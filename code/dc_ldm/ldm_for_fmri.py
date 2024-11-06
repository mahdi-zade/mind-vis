import numpy as np
import wandb
import torch
from dc_ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import os
from dc_ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sc_mbm.mae_for_eeg import eeg_encoder  # Replace with EEG encoder

def create_model_from_config(config, num_channels, global_pool):
    model = eeg_encoder(num_channels=num_channels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                        depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool)
    return model

class cond_stage_model(nn.Module):
    def __init__(self, metafile, num_channels, cond_dim=1280, global_pool=True):
        super().__init__()
        # Prepare pretrained EEG MAE
        model = create_model_from_config(metafile['config'], num_channels, global_pool)
        model.load_checkpoint(metafile['model'])
        self.eeg_seq_len = model.num_patches
        self.eeg_latent_dim = model.embed_dim
        if not global_pool:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.eeg_seq_len, self.eeg_seq_len // 2, 1, bias=True),
                nn.Conv1d(self.eeg_seq_len // 2, 77, 1, bias=True)
            )
        self.dim_mapper = nn.Linear(self.eeg_latent_dim, cond_dim, bias=True)
        self.global_pool = global_pool

    def forward(self, x):
        latent_crossattn = self.mae(x)  # Encoder output for EEG
        if not self.global_pool:
            latent_crossattn = self.channel_mapper(latent_crossattn)
        latent_crossattn = self.dim_mapper(latent_crossattn)
        return latent_crossattn

class fLDM:

    def __init__(self, metafile, num_channels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/ldm/label2img',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=True):
        self.ckp_path = os.path.join(pretrain_root, 'model.ckpt')
        self.config_path = os.path.join(pretrain_root, 'config.yaml')
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim

        model = instantiate_from_config(config.model)
        pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']

        m, u = model.load_state_dict(pl_sd, strict=False)
        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(metafile, num_channels, self.cond_dim, global_pool=global_pool)

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        self.device = device
        self.model = model
        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.eeg_latent_dim = model.cond_stage_model.eeg_latent_dim
        self.metafile = metafile

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        self.model.run_full_validation_threshold = 0.15

        print('\n##### Stage One: only optimize conditional encoders #####')
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        self.model.unfreeze_whole_model()

        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint.pth')
        )

    @torch.no_grad()
    def generate(self, eeg_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None):
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels,
                     self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                     HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)

        if state is not None:
            torch.cuda.set_rng_state(state)

        with model.ema_scope():
            model.eval()
            for count, item in enumerate(eeg_embedding):
                if limit is not None and count >= limit:
                    break
                latent = item['eeg']  # Adjust for EEG
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w')  # h w c

                print(f"Rendering {num_samples} examples in {ddim_steps} steps.")

                c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                  conditioning=c,
                                                  batch_size=num_samples,
                                                  shape=shape,
                                                  verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image + 1.0) / 2.0, min=0.0, max=1.0)

                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0))  # put groundtruth at first

        # Display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples + 1)

        # Convert to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')

        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)
