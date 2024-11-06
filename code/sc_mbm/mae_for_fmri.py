import sc_mbm.utils as ut
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
import torch.nn.functional as F
class PatchEmbed1D(nn.Module):
    """ This class is designed to process EEG data and convert it into "patches" 
    which are then transformed into embeddings using a 1D convolutional layer.
    """
    def __init__(self, num_channels=32, seq_length=256, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = seq_length // patch_size
        self.patch_shape = patch_size
        self.num_channels = num_channels
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        """The size of the kernel (filter) used in the convolution is set to the patch size, which means the filter will cover exactly one patch at a time. 
           The stride of the convolution, which defines how far the filter moves across the input is set to the patch size, so the filter moves exactly by the length of each patch, effectively extracting one patch at a time.
        """
    def forward(self, input, **kwargs):
        embedded_input = self.conv_layer(input).transpose(1, 2).contiguous()  # The shape changes from (B, embed_dim, num_patches) to (B, num_patches, embed_dim).
        # .contiguous(): This makes sure that the resulting tensor is stored in a contiguous block of memory, which is required for further operations in PyTorch.
        return embedded_input

class MAEforFMRI(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, num_channels=32, seq_length=256, patch_size=16, embed_dim=1024, in_chans=1,
                 depth=24, num_heads=16, decoder_embed_dim=512,
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, focus_range=None, focus_rate=None, img_recon_weight=1.0,
                 use_nature_img_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed1D(num_channels, seq_length, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * in_chans, bias=True)  # encoder to decoder
        # --------------------------------------------------------------------------

        # nature image decoder specifics
        if use_nature_img_loss:
            self.nature_img_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.nature_img_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.nature_img_decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

            self.nature_img_decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(2)])

            self.nature_img_decoder_norm = norm_layer(decoder_embed_dim)
            self.nature_img_decoder_pred = nn.Sequential(
                nn.Conv1d(num_patches, 512, kernel_size=1, stride=1, bias=True),
                nn.Linear(decoder_embed_dim, 28 * 28, bias=True)
            )
            # --------------------------------------------------------------------------

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.focus_range = focus_range
        self.focus_rate = focus_rate
        self.img_recon_weight = img_recon_weight
        self.use_nature_img_loss = use_nature_img_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        if self.use_nature_img_loss:
            nature_img_decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.nature_img_decoder_pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
            self.nature_img_decoder_pos_embed.data.copy_(torch.from_numpy(nature_img_decoder_pos_embed).float().unsqueeze(0))
            torch.nn.init.normal_(self.nature_img_mask_token, std=.02)

        # initialize patch_embed
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                

    def random_masking(self, sequence_data, mask_ratio):
        """
        Perform random masking for each sample.

        Parameters:
        - sequence_data: [batch_size, sequence_length, embedding_dim]
            - batch_size: Number of samples
            - sequence_length: Total length of the sequence (e.g., EEG time points or channels)
            - embedding_dim: Dimensionality of embeddings per patch

        This function applies random masking to segments of the input data based on the mask_ratio
        to save computational power. 
        Due to spatial redundancy across EEG channels, critical information can still be recovered 
        even if portions of the data are masked, as nearby signals contain overlapping information.

        Returns:
        - masked_data: Masked version of the input data
        - mask: Binary mask indicating which parts were kept (1) and which were masked (0)
        - restore_indices: Indices to reverse the shuffling order during masking
        """
        batch_size, sequence_length, embedding_dim = sequence_data.shape
        num_keep = int(sequence_length * (1 - mask_ratio))

        if self.focus_range is not None:
            num_mask = sequence_length - num_keep
            mask_weights = [1 - self.focus_rate] * sequence_length
            mask_weights[self.focus_range[0] // self.patch_size : self.focus_range[1] // self.patch_size] = \
                [self.focus_rate] * (self.focus_range[1] // self.patch_size - self.focus_range[0] // self.patch_size)
            mask_weights = torch.tensor(mask_weights).repeat(batch_size, 1).to(sequence_data.device)
            focused_mask_indices = torch.multinomial(mask_weights, num_mask, replacement=False)

        random_noise = torch.rand(batch_size, sequence_length, device=sequence_data.device)  # Random values [0, 1]
        if self.focus_range is not None:
            for i in range(batch_size):
                random_noise[i, focused_mask_indices[i, :]] = 1.1  # Mark selected indices for masking

        shuffled_indices = torch.argsort(random_noise, dim=1)  # Sort: smallest values are kept
        restore_indices = torch.argsort(shuffled_indices, dim=1)  # To reverse the shuffle order

        keep_indices = shuffled_indices[:, :num_keep]
        masked_data = torch.gather(sequence_data, 1, shuffled_indices)

        mask = torch.zeros((batch_size, sequence_length), device=sequence_data.device)
        mask[:, keep_indices] = 1.0
        return masked_data, mask, restore_indices

    def forward_encoder(self, input_data, mask_ratio):
        """
        Encoder forward pass.

        Parameters:
        - input_data: Original input data (e.g., EEG sequence)
        - mask_ratio: Ratio of patches to mask

        Returns:
        - latent representation, mask, and restoration indices
        """
        # Divide input data into patches
        embedded_patches = self.patch_embed(input_data)
        # Add positional encoding (no classification token)
        # Positional information is added to the patches to give the model a sense of the order of data points.
        embedded_patches = embedded_patches + self.pos_embed[:, 1:, :]

        # Apply random masking. Some patches are randomly masked, with a given mask_ratio. The masked patches help the model learn how to reconstruct missing data.
        masked_patches, mask, restore_indices = self.random_masking(embedded_patches, mask_ratio)

        # Add classification token with position embedding. A special token (usually representing the class or "identity" of the data) is added to the beginning of the sequence.
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(masked_patches.shape[0], -1, -1)
        masked_patches = torch.cat((cls_tokens, masked_patches), dim=1)

        # Pass through Transformer blocks
        # The patches, now with the classification token, pass through several Transformer layers to capture complex relationships between them.
        for block in self.blocks:
            masked_patches = block(masked_patches)
        return self.norm(masked_patches), mask, restore_indices

    def forward_decoder(self, latent_representation, restore_indices):
        """
        Decoder forward pass.

        Parameters:
        - latent_representation: Masked data output from encoder
        - restore_indices: Indices to reorder masked tokens

        Returns:
        - reconstructed data
        """
        decoded_patches = self.decoder_embed(latent_representation)

        # Prepare mask tokens for missing segments
        mask_tokens = self.mask_token.repeat(decoded_patches.shape[0], restore_indices.shape[1] + 1 - decoded_patches.shape[1], 1)
        reordered_patches = torch.cat([decoded_patches[:, 1:, :], mask_tokens], dim=1)
        reordered_patches = torch.gather(reordered_patches, dim=1, index=restore_indices.unsqueeze(-1).repeat(1, 1, decoded_patches.shape[2]))
        
        decoded_patches = torch.cat([decoded_patches[:, :1, :], reordered_patches], dim=1)  # Add classification token

        # Add positional encoding
        decoded_patches = decoded_patches + self.decoder_pos_embed

        # Pass through decoder Transformer blocks
        for block in self.decoder_blocks:
            decoded_patches = block(decoded_patches)
        return self.decoder_pred(self.decoder_norm(decoded_patches))[:, 1:, :]

    def forward_loss(self, original_data, predictions, mask):
        """
        Calculate loss based on the difference between the original and reconstructed data.

        Parameters:
        - original_data: Original data before patching
        - predictions: Reconstructed data from decoder
        - mask: Mask indicating which patches were masked

        Returns:
        - Loss calculated only over masked patches
        """
        target_patches = ut.patchify(original_data)

        loss = (predictions - target_patches) ** 2
        loss = loss.mean(dim=-1)  # Mean loss per patch

        # Calculate average loss over masked patches
        return (loss * mask).sum() / mask.sum() if mask.sum() != 0 else (loss * mask).sum()

    def forward(self, eeg_data, img_features=None, valid_idx=None, mask_ratio=0.75):
        """
        Complete forward pass.

        Parameters:
        - eeg_data: Input EEG data
        - img_features: Features from nature images (optional)
        - valid_idx: Indices for valid samples (optional)
        - mask_ratio: Ratio of patches to mask

        Returns:
        - Loss, prediction, and mask
        """
        latent, mask, restore_indices = self.forward_encoder(eeg_data, mask_ratio)
        predictions = self.forward_decoder(latent, restore_indices)
        total_loss = self.forward_loss(eeg_data, predictions, mask)

        if self.use_nature_img_loss and img_features is not None and valid_idx is not None:
            # Forward pass through additional decoder for nature image reconstruction
            recon_nature_image = self.forward_nature_img_decoder(latent[valid_idx], restore_indices[valid_idx])
            loss_nature_img = self.forward_nature_img_loss(img_features, recon_nature_image)

            if torch.isnan(loss_nature_img).any():
                print("loss_nature_img contains NaN values")

            total_loss += self.img_recon_weight * loss_nature_img

        return total_loss, predictions, mask

class eeg_encoder(nn.Module):
    def __init__(self, num_channels=32, time_points=1000, patch_size=50, embed_dim=1024,
                 depth=24, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, global_pool=True):
        super().__init__()
        self.patch_embed = PatchEmbed1D(time_points, patch_size, num_channels, embed_dim)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # Fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed_dim = embed_dim

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.global_pool = global_pool
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize and freeze pos_embed using sin-cos embedding
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        # Embed patches
        x = self.patch_embed(x)

        # Add positional embedding without cls token
        x = x + self.pos_embed[:, 1:, :]
        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x.mean(dim=1, keepdim=True)
        x = self.norm(x)

        return x

    def forward(self, eeg_data):
        if eeg_data.ndim == 2:
            eeg_data = torch.unsqueeze(eeg_data, dim=0)  # N, num_channels, time_points
        latent = self.forward_encoder(eeg_data)  # N, n_seq, embed_dim
        return latent  # N, n_seq, embed_dim

    def load_checkpoint(self, state_dict):
        if self.global_pool:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k and 'norm' not in k)}
        else:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
        ut.interpolate_pos_embed(self, state_dict)

        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        print('unexpected keys:', m)
        return
