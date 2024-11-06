import numpy as np
import math
import torch
import os

def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_l = np.arange(length, dtype=np.float32)

    grid_l = grid_l.reshape([1, length])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_l)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # cls token
        # height (== width) for the checkpoint position embedding
        orig_size = int(pos_embed_checkpoint.shape[-2] - num_extra_tokens)
        # height (== width) for the new position embedding
        new_size = int(num_patches)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %d to %d" % (orig_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, embedding_size).permute(0, 2, 1)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size))
            pos_tokens = pos_tokens.permute(0, 2, 1)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed



def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config.warmup_epochs:
        lr = config.lr * epoch / config.warmup_epochs 
    else:
        lr = config.min_lr + (config.lr - config.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config.warmup_epochs) / (config.num_epoch - config.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def save_model(config, epoch, model, optimizer, loss_scaler, checkpoint_paths):
    os.makedirs(checkpoint_paths, exist_ok=True)
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'config': config,
    }
    torch.save(to_save, os.path.join(checkpoint_paths, 'checkpoint.pth'))
    

def load_model(config, model, checkpoint_path ):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print(f'Model loaded with {checkpoint_path}')

def patchify(self, input_data):
    """
    input_data: (batch_size, num_channels, sequence_length)
        - batch_size: The number of samples
        - num_channels: The number of channels
        - sequence_length: The length of the sequence or time dimension
        
    Returns:
    patches: (batch_size, num_patches, patch_size)
    
    This function splits the input data into patches along the sequence dimension.
    By reshaping the input into smaller segments, patchify establishes a high-capacity 
    representation space. This can increase the information capacity by using a larger 
    embedding-to-patch-size ratio.
    """
    patch_size = self.patch_embed.patch_size
    
    # Ensure the sequence length is evenly divisible by the patch size
    assert input_data.ndim == 3 and input_data.shape[2] % patch_size == 0

    # Calculate the number of patches along the sequence dimension
    num_patches = input_data.shape[2] // patch_size
    
    # Reshape input data to create patches of size (patch_size) along the sequence dimension
    patches = input_data.reshape(shape=(input_data.shape[0], num_patches, patch_size))
    
    return patches


def unpatchify(self, patch_sequence):
    """
    patch_sequence: (batch_size, num_patches, patch_size)
        - batch_size: The number of samples
        - num_patches: The number of patches along the sequence dimension
        - patch_size: The size of each patch
        
    Returns:
    reconstructed_data: (batch_size, num_channels, sequence_length)
        - num_channels: Restored to 1 (since it was split into patches)
        - sequence_length: The length of the original sequence (num_patches * patch_size)
        
    This function reverses the patchify operation, reconstructing the original
    data by combining patches back into the full sequence along the time dimension.
    """
    patch_size = self.patch_embed.patch_size
    num_patches = patch_sequence.shape[1]
    
    # Reshape patches to reconstruct the original sequence length
    reconstructed_data = patch_sequence.reshape(shape=(patch_sequence.shape[0], 1, num_patches * patch_size))
    
    return reconstructed_data
