import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# We will import these once they are adapted or if their existing structure is sufficient
# For now, define dummy classes or assume they will be available.
# from video_modality import VideoModality
# from audio_modality import AudioModality
# from shared_backbone import SharedTransformerBackbone, MAEDecoder

# Import actual components
from video_modality import VideoModality
from audio_modality import AudioModality
from shared_backbone import SharedTransformerBackbone, MAEDecoder, ProjectionHead, SoundClassificationHead

# Placeholder for actual imports - replace with real ones when available and adapted
class PlaceholderVideoModality(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Example: store relevant config like ViT hidden size and patch pixel dim
        self.vit_hidden_dim = kwargs.get('vit_hidden_dim', 768)
        self.patch_pixel_dim = kwargs.get('patch_pixel_dim', 16*16*3)
        self.num_total_patches = kwargs.get('num_total_patches', 16 * (224//16)**2) # e.g. 16 frames * 196 patches/frame
        print(f"PlaceholderVideoModality: vit_hidden_dim={self.vit_hidden_dim}, patch_pixel_dim={self.patch_pixel_dim}, num_total_patches={self.num_total_patches}")


    def get_patches_for_mae(self, video_paths_batch, device):
        B = len(video_paths_batch)
        # Simulate output: all_patch_embeddings, original_patches_pixels, N_total
        all_patch_embeddings = torch.randn(B, self.num_total_patches, self.vit_hidden_dim, device=device)
        original_patches_pixels = torch.randn(B, self.num_total_patches, self.patch_pixel_dim, device=device)
        return all_patch_embeddings, original_patches_pixels, self.num_total_patches

class PlaceholderAudioModality(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.vit_hidden_dim = kwargs.get('vit_hidden_dim', 768)
        self.patch_spectrogram_dim = kwargs.get('patch_spectrogram_dim', 16*16*1)
        self.num_total_patches = kwargs.get('num_total_patches', (224//16)**2) # e.g. 196 patches for a spectrogram
        print(f"PlaceholderAudioModality: vit_hidden_dim={self.vit_hidden_dim}, patch_spectrogram_dim={self.patch_spectrogram_dim}, num_total_patches={self.num_total_patches}")

    def get_patches_for_mae(self, video_paths_batch, device): # Audio also comes from video files
        B = len(video_paths_batch)
        all_patch_embeddings = torch.randn(B, self.num_total_patches, self.vit_hidden_dim, device=device)
        original_spectrogram_patches = torch.randn(B, self.num_total_patches, self.patch_spectrogram_dim, device=device)
        return all_patch_embeddings, original_spectrogram_patches, self.num_total_patches

# Assuming shared_backbone.py is in the same directory and contains these
try:
    from shared_backbone import SharedTransformerBackbone, MAEDecoder
except ImportError:
    print("Warning: shared_backbone.py not found. Using placeholder MAE components.")
    class SharedTransformerBackbone(nn.Module): # Placeholder
        def __init__(self, embed_dim, **kwargs):
            super().__init__()
            self.embed_dim = embed_dim
            self.transformer_encoder = nn.Identity() # Simple placeholder
            self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
            print(f"PlaceholderSharedTransformerBackbone with embed_dim={embed_dim}")
        def forward(self, src_tokens, **kwargs):
            B, N, D = src_tokens.shape
            cls = self.cls_token.expand(B, -1, -1)
            return torch.cat([cls, src_tokens], dim=1) # crude simulation

    class MAEDecoder(nn.Module): # Placeholder
        def __init__(self, output_patch_dim, embed_dim, **kwargs):
            super().__init__()
            self.output_patch_dim = output_patch_dim
            self.embed_dim = embed_dim
            print(f"PlaceholderMAEDecoder: output_patch_dim={output_patch_dim}, embed_dim={embed_dim}")
        def forward(self, unmasked_features_from_backbone, unmasked_indices, masked_indices, total_patches_in_sequence):
            B, N_unmasked, D_embed = unmasked_features_from_backbone.shape
            # Simulate reconstruction of all patches
            return torch.randn(B, total_patches_in_sequence, self.output_patch_dim, device=unmasked_features_from_backbone.device)


def random_masking_indices(B, N_total, mask_ratio, device):
    """
    Generates indices for random masking.
    Args:
        B: Batch size
        N_total: Total number of patches in the sequence.
        mask_ratio: Proportion of patches to mask.
        device: Torch device.
    Returns:
        unmasked_indices: (B, N_unmasked) - Original indices of unmasked patches.
        masked_indices: (B, N_masked) - Original indices of masked patches.
        N_unmasked: Number of unmasked patches.
    """
    N_unmasked = int(N_total * (1 - mask_ratio))
    if N_unmasked == 0 and N_total > 0 : # Ensure at least one patch is unmasked if possible
        N_unmasked = 1 
    if N_unmasked > N_total: # Should not happen with typical mask_ratio < 1
        N_unmasked = N_total

    noise = torch.rand(B, N_total, device=device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    
    unmasked_indices = ids_shuffle[:, :N_unmasked]
    masked_indices = ids_shuffle[:, N_unmasked:] # The rest are masked
    
    # Sort for consistency if needed by downstream logic (MAE typically uses scatter)
    unmasked_indices = torch.sort(unmasked_indices, dim=1)[0]
    masked_indices = torch.sort(masked_indices, dim=1)[0]
    
    return unmasked_indices, masked_indices, N_unmasked

class T3_AV_Model(nn.Module):
    def __init__(self, 
                 video_modality_params: dict, # Params for VideoModality constructor
                 audio_modality_params: dict, # Params for AudioModality constructor
                 shared_backbone_params: dict, # Params for SharedTransformerBackbone (num_heads, num_layers, etc.)
                                               # embed_dim will be derived from modality encoders
                 video_mae_decoder_base_params: dict, # Base params for MAEDecoder (num_heads, num_layers, etc.)
                                                      # embed_dim & output_patch_dim will be derived
                 audio_mae_decoder_base_params: dict, # Similar to video_mae_decoder_base_params
                 proj_head_base_params: dict, # Params for ProjectionHead (hidden_dim, etc.)
                 sound_clf_head_params: dict, # Params for SoundClassificationHead (hidden_dim, num_classes)
                 contrastive_embed_dim: int = 128, # Dimension for contrastive loss
                 contrastive_temperature: float = 0.07, # Temperature for InfoNCE
                 mae_mask_ratio=0.75):
        super().__init__()
        self.mae_mask_ratio = mae_mask_ratio
        self.temperature = contrastive_temperature

        # 1. Instantiate Modality Encoders
        self.video_encoder = VideoModality(**video_modality_params)
        self.audio_encoder = AudioModality(**audio_modality_params)
        print("Actual VideoModality and AudioModality instantiated.")

        # Ensure both encoders use the same hidden dimension from their ViTs for shared backbone compatibility
        assert self.video_encoder.vit_hidden_size == self.audio_encoder.vit_hidden_size, \
            "Video and Audio ViT hidden sizes must match for shared backbone input."
        self.modal_embed_dim = self.video_encoder.vit_hidden_size # This will be embed_dim for shared_backbone

        # 2. Configure and Instantiate Shared Backbone
        # Max sequence length for shared backbone's positional encoding is the max number of *unmasked* patches + 1 (for CLS)
        # from either modality. This is a bit dynamic, so set a sufficiently large max_len for pos_encoder in SharedBackbone,
        # or calculate based on max possible unmasked patches.
        # Let's use the max total patches from either modality initially for simplicity in SharedBackbone's PositionalEncoding max_len.
        # The actual sequence length passed to shared_backbone.forward will be num_unmasked + 1.
        max_possible_unmasked_video = int(self.video_encoder.mae_total_patches * (1 - self.mae_mask_ratio)) +1
        max_possible_unmasked_audio = int(self.audio_encoder.mae_num_patches * (1 - self.mae_mask_ratio)) +1
        
        _shared_backbone_params = {
            'embed_dim': self.modal_embed_dim,
            'max_seq_len': max(max_possible_unmasked_video, max_possible_unmasked_audio, 256), # Ensure a min length too
            **shared_backbone_params # num_heads, num_layers, mlp_dim, dropout
        }
        self.shared_backbone = SharedTransformerBackbone(**_shared_backbone_params)
        print("Actual SharedTransformerBackbone instantiated.")

        # 3. Configure and Instantiate MAE Decoders
        _video_mae_decoder_params = {
            'embed_dim': self.modal_embed_dim,
            'output_patch_dim': self.video_encoder.mae_patch_pixel_dim,
            'max_total_patches': self.video_encoder.mae_total_patches,
            **video_mae_decoder_base_params # num_heads, num_decoder_layers, mlp_dim, dropout
        }
        self.video_mae_decoder = MAEDecoder(**_video_mae_decoder_params)

        _audio_mae_decoder_params = {
            'embed_dim': self.modal_embed_dim,
            'output_patch_dim': self.audio_encoder.mae_patch_spectrogram_dim,
            'max_total_patches': self.audio_encoder.mae_num_patches,
            **audio_mae_decoder_base_params
        }
        self.audio_mae_decoder = MAEDecoder(**_audio_mae_decoder_params)
        print("Actual MAEDecoders instantiated.")
        
        # 4. Configure and Instantiate Projection Heads (for Stage 2)
        _proj_head_params = {
            'input_dim': self.modal_embed_dim,
            'output_dim': contrastive_embed_dim,
            **proj_head_base_params # hidden_dim, use_gelu
        }
        self.video_projection_head = ProjectionHead(**_proj_head_params)
        self.audio_projection_head = ProjectionHead(**_proj_head_params)
        print("ProjectionHeads instantiated.")
        
        # 5. Configure and Instantiate Sound Classification Head (for Fine-tuning)
        _sound_clf_head_params_updated = {
            'input_dim': self.modal_embed_dim * 2, # Fused features from audio and video CLS tokens
            **sound_clf_head_params # hidden_dim, num_classes, use_gelu (num_classes must be provided)
        }
        if 'num_classes' not in _sound_clf_head_params_updated or _sound_clf_head_params_updated['num_classes'] <= 0:
            raise ValueError("SoundClassificationHead requires 'num_classes' > 0 in sound_clf_head_params")
        self.sound_classifier = SoundClassificationHead(**_sound_clf_head_params_updated)
        print(f"SoundClassificationHead instantiated for {_sound_clf_head_params_updated['num_classes']} classes, expecting input_dim={self.modal_embed_dim * 2}.")
        
        print("T3_AV_Model initialized for Stages 1, 2 & Fine-tuning with AV fusion.")

    def _forward_mae_modality(self, modality_name, all_patches_embed, original_patches_target, 
                              N_total_patches, mae_decoder, device):
        B, _, D_embed = all_patches_embed.shape

        unmasked_indices, masked_indices, N_unmasked = \
            random_masking_indices(B, N_total_patches, self.mae_mask_ratio, device=device)
        
        if N_unmasked == 0:
            # Should not happen if random_masking_indices ensures at least one unmasked
            print(f"Warning: MAE for {modality_name} has no unmasked patches for batch size {B}, N_total {N_total_patches}. Returning zero loss.")
            return torch.tensor(0.0, device=device, requires_grad=True)

        idx_expand = unmasked_indices.unsqueeze(-1).expand(-1, -1, D_embed)
        unmasked_embed_gathered = torch.gather(all_patches_embed, dim=1, index=idx_expand)
        
        backbone_out = self.shared_backbone(unmasked_embed_gathered) 
        unmasked_backbone_features = backbone_out[:, 1:] 

        reconstructed_all_patches = mae_decoder(
            unmasked_features_from_backbone=unmasked_backbone_features,
            unmasked_indices=unmasked_indices, 
            masked_indices=masked_indices,     
            total_patches_in_sequence=N_total_patches
        )
        
        if masked_indices.numel() == 0: # No patches were masked
             print(f"Warning: MAE for {modality_name} has no masked patches to compute loss on. Returning zero loss.")
             return torch.tensor(0.0, device=device, requires_grad=True)

        idx_expand_target = masked_indices.unsqueeze(-1).expand(-1, -1, original_patches_target.shape[-1])
        target_masked_patches = torch.gather(original_patches_target, dim=1, index=idx_expand_target)
        
        idx_expand_recon = masked_indices.unsqueeze(-1).expand(-1, -1, reconstructed_all_patches.shape[-1])
        pred_masked_patches = torch.gather(reconstructed_all_patches, dim=1, index=idx_expand_recon)

        loss = F.mse_loss(pred_masked_patches, target_masked_patches)
        return loss

    def forward_video_mae(self, video_paths_batch, device):
        # This now uses the actual self.video_encoder
        # Expects get_patches_for_mae to return: (all_patch_embeds, original_pixels, N_total)
        # The batching should happen here or in the dataset. For now, assume video_paths_batch is a list.
        
        batch_all_embeds = []
        batch_original_pixels = []
        current_N_v_total = None # Should be same for all in a consistent setup

        for video_path in video_paths_batch:
            all_video_patches_embed, original_video_patches_pixels, N_v_total_single = \
                self.video_encoder.get_patches_for_mae(video_path, device=device)
            batch_all_embeds.append(all_video_patches_embed) # each is (1, N_v, D)
            batch_original_pixels.append(original_video_patches_pixels) # each is (1, N_v, P_dim)
            if current_N_v_total is None: current_N_v_total = N_v_total_single
            else: assert current_N_v_total == N_v_total_single, "Mismatch in N_v_total within batch"

        if not batch_all_embeds: # Should not happen if video_paths_batch is not empty
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        # Concatenate batch items: list of (1, N, D) -> (B, N, D)
        all_video_patches_embed_batch = torch.cat(batch_all_embeds, dim=0)
        original_video_patches_pixels_batch = torch.cat(batch_original_pixels, dim=0)
        
        return self._forward_mae_modality(
            modality_name="video",
            all_patches_embed=all_video_patches_embed_batch,
            original_patches_target=original_video_patches_pixels_batch,
            N_total_patches=current_N_v_total,
            mae_decoder=self.video_mae_decoder,
            device=device
        )

    def forward_audio_mae(self, video_paths_batch, device):
        batch_all_embeds = []
        batch_original_spectrograms = []
        current_N_a_total = None

        for video_path in video_paths_batch:
            all_audio_patches_embed, original_audio_spectrogram_patches, N_a_total_single = \
                self.audio_encoder.get_patches_for_mae(video_path, device=device)
            batch_all_embeds.append(all_audio_patches_embed)
            batch_original_spectrograms.append(original_audio_spectrogram_patches)
            if current_N_a_total is None: current_N_a_total = N_a_total_single
            else: assert current_N_a_total == N_a_total_single, "Mismatch in N_a_total within batch"

        if not batch_all_embeds:
            return torch.tensor(0.0, device=device, requires_grad=True)

        all_audio_patches_embed_batch = torch.cat(batch_all_embeds, dim=0)
        original_audio_spectrogram_patches_batch = torch.cat(batch_original_spectrograms, dim=0)

        return self._forward_mae_modality(
            modality_name="audio",
            all_patches_embed=all_audio_patches_embed_batch,
            original_patches_target=original_audio_spectrogram_patches_batch,
            N_total_patches=current_N_a_total,
            mae_decoder=self.audio_mae_decoder,
            device=device
        )

    def forward_stage1(self, video_paths_batch, device, train_video_mae=True, train_audio_mae=True):
        total_loss = torch.tensor(0.0, device=device)
        video_mae_loss_val = None
        audio_mae_loss_val = None

        if not video_paths_batch: # Handle empty batch explicitly
            print("Warning: forward_stage1 received an empty batch.")
            return total_loss, video_mae_loss_val, audio_mae_loss_val

        if train_video_mae:
            video_mae_loss = self.forward_video_mae(video_paths_batch, device)
            total_loss += video_mae_loss
            video_mae_loss_val = video_mae_loss.item()
        
        if train_audio_mae:
            audio_mae_loss = self.forward_audio_mae(video_paths_batch, device)
            total_loss += audio_mae_loss
            audio_mae_loss_val = audio_mae_loss.item()
        
        return total_loss, video_mae_loss_val, audio_mae_loss_val

    # --- Stage 2 Methods --- 
    def _compute_infonce_loss(self, z_a, z_v, temperature):
        """ Computes InfoNCE loss for one direction (e.g., a -> v) """
        # z_a, z_v are L2 normalized embeddings (B, D_contrastive)
        logits = torch.mm(z_a, z_v.t()) / temperature # (B, B)
        labels = torch.arange(logits.shape[0], device=logits.device) # Positive pairs are on diagonal
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward_stage2(self, video_paths_batch, device):
        """
        Main forward pass for Stage 2 Contrastive Learning.
        Args:
            video_paths_batch: A list or batch of video file paths.
            device: Torch device.
        Returns:
            contrastive_loss: Scalar tensor.
            batch_size_processed: Number of samples successfully processed for both modalities.
        """
        # 1. Get features from modality encoders
        # These forward methods now return (B_proc, N_patches, D_modal)
        video_features = self.video_encoder(video_paths_batch, device=device)
        audio_features = self.audio_encoder(video_paths_batch, device=device)

        # 2. Handle potential batch size mismatches due to errors in encoders
        # If B_v != B_a, we cannot compute standard contrastive loss. 
        # For now, assert they match. Robust handling might involve finding common successful indices.
        if video_features.shape[0] != audio_features.shape[0]:
            print(f"Warning: Batch size mismatch between video ({video_features.shape[0]}) and audio ({audio_features.shape[0]}) encoders. Skipping contrastive loss calculation for this batch.")
            return torch.tensor(0.0, device=device, requires_grad=True), 0
        
        batch_size_processed = video_features.shape[0]
        if batch_size_processed == 0:
            # print("Warning: forward_stage2 resulted in zero processed samples.")
            return torch.tensor(0.0, device=device, requires_grad=True), 0

        # 3. Pass through shared backbone
        # TODO: Handle padding masks if modality encoders return variable length sequences? 
        # For now, assume fixed sequence lengths (total_video_patches, num_audio_patches)
        bb_out_v = self.shared_backbone(video_features) # (B, N_v+1, D_modal)
        bb_out_a = self.shared_backbone(audio_features) # (B, N_a+1, D_modal)

        # 4. Pass through projection heads (using CLS token)
        z_v = self.video_projection_head(bb_out_v, use_cls_token=True) # (B, D_contrastive)
        z_a = self.audio_projection_head(bb_out_a, use_cls_token=True) # (B, D_contrastive)

        # 5. L2 Normalize embeddings
        z_v = F.normalize(z_v, dim=1, p=2)
        z_a = F.normalize(z_a, dim=1, p=2)

        # 6. Compute symmetric InfoNCE loss
        loss_a_v = self._compute_infonce_loss(z_a, z_v, self.temperature)
        loss_v_a = self._compute_infonce_loss(z_v, z_a, self.temperature)
        total_contrastive_loss = (loss_a_v + loss_v_a) / 2
        
        return total_contrastive_loss, batch_size_processed

    # --- Fine-tuning Methods --- 
    def forward_finetune(self, video_paths_batch, labels_batch, device):
        """
        Forward pass for fine-tuning on sound classification using audio-visual fusion.
        Args:
            video_paths_batch: List of video file paths.
            labels_batch: Tensor of ground truth label indices (B).
            device: Torch device.
        Returns:
            loss: Scalar classification loss (CrossEntropy).
            logits: Tensor of output logits (B, num_classes).
            processed_batch_size: int, number of samples successfully processed by both modalities.
        """
        # 1. Get features from modality encoders
        audio_features = self.audio_encoder(video_paths_batch, device=device) # (B_proc_a, N_a, D_modal)
        video_features = self.video_encoder(video_paths_batch, device=device) # (B_proc_v, N_v, D_modal)

        bsz_a = audio_features.shape[0]
        bsz_v = video_features.shape[0]
        num_classes = self.sound_classifier.fc2.out_features

        # 2. Handle batch size mismatches or empty batches from encoders
        if bsz_a == 0 or bsz_v == 0:
            print(f"Warning: Fine-tune encoders returned 0 samples (Audio: {bsz_a}, Video: {bsz_v}). Skipping batch.")
            dummy_logits = torch.zeros(len(video_paths_batch), num_classes, device=device)
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return dummy_loss, dummy_logits, 0

        if bsz_a != bsz_v:
            print(f"Warning: Fine-tune batch size mismatch (Audio: {bsz_a}, Video: {bsz_v}). This should ideally be handled by aligning inputs before this function. Skipping batch.")
            # This scenario is tricky. For now, we skip if they don't match after encoder processing.
            # A robust solution might involve pre-filtering or aligning indices of successfully processed items.
            dummy_logits = torch.zeros(len(video_paths_batch), num_classes, device=device)
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return dummy_loss, dummy_logits, 0 # Or min(bsz_a, bsz_v) if we were to take a subset
        
        # At this point, bsz_a == bsz_v == processed_batch_size
        processed_batch_size = bsz_a

        # Ensure labels match the processed batch size
        if processed_batch_size != labels_batch.shape[0]:
            print(f"Warning: Fine-tune label batch size ({labels_batch.shape[0]}) does not match processed feature batch size ({processed_batch_size}). This implies an issue upstream. Skipping batch.")
            dummy_logits = torch.zeros(labels_batch.shape[0], num_classes, device=device)
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return dummy_loss, dummy_logits, 0

        # 3. Pass features through shared backbone
        bb_out_a = self.shared_backbone(audio_features) # (B_proc, N_a+1, D_modal)
        bb_out_v = self.shared_backbone(video_features) # (B_proc, N_v+1, D_modal)
        
        # 4. Extract CLS tokens
        cls_a = bb_out_a[:, 0] # (B_proc, D_modal)
        cls_v = bb_out_v[:, 0] # (B_proc, D_modal)
        
        # 5. Fuse CLS tokens by concatenation
        fused_cls_features = torch.cat((cls_a, cls_v), dim=1) # (B_proc, 2 * D_modal)
        
        # 6. Get classification logits from the sound classifier head
        # The SoundClassificationHead.forward now handles 2D input directly
        logits = self.sound_classifier(fused_cls_features) # (B_proc, num_classes)
        
        # 7. Calculate cross-entropy loss
        loss = F.cross_entropy(logits, labels_batch) # labels_batch should match B_proc
        
        return loss, logits, processed_batch_size


if __name__ == '__main__':
    print("\n--- Testing T3_AV_Model Stages 1, 2 & Finetune with AV Fusion ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configurations for actual modules (Reduced for test) ---
    pretrained_vit = 'google/vit-base-patch16-224-in21k' 
    video_params = {
        'num_frames_to_sample': 4, 
        'pretrained_vit_name': pretrained_vit,
        'output_embed_dim': 512 
    }
    audio_params = {
        'sample_rate': 16000, 'n_mels': 128, 'n_fft': 1024, 'hop_length': 512,
        'target_spectrogram_height': 224, 'target_spectrogram_width': 224,
        'pretrained_vit_name': pretrained_vit, 'output_embed_dim': 512, 
        'audio_duration_secs': 2 
    }
    shared_bb_base_params = {
        'num_heads': 8, 'num_layers': 3, 'mlp_dim': 768 * 2, 'dropout': 0.1
    }
    mae_dec_base_params = {
        'num_heads': 4, 'num_decoder_layers': 1, 'mlp_dim': 768 * 2, 'dropout': 0.1
    }
    proj_head_base_params = {
        'hidden_dim': 768, 
        'use_gelu': True
    }
    # New configs for Fine-tuning
    sound_clf_head_params_for_test = {
        'hidden_dim': 512,     
        'num_classes': 10,      
        'use_gelu': True
        # input_dim will be set by T3_AV_Model __init__ based on modal_embed_dim * 2
    }
    contrastive_dim = 128
    contrastive_temp = 0.07

    # Instantiate the main model
    try:
        t3_model = T3_AV_Model(
            video_modality_params=video_params,
            audio_modality_params=audio_params,
            shared_backbone_params=shared_bb_base_params,
            video_mae_decoder_base_params=mae_dec_base_params,
            audio_mae_decoder_base_params=mae_dec_base_params,
            proj_head_base_params=proj_head_base_params, 
            sound_clf_head_params=sound_clf_head_params_for_test, 
            contrastive_embed_dim=contrastive_dim,
            contrastive_temperature=contrastive_temp,
            mae_mask_ratio=0.75
        ).to(device)

        # Create/find dummy video paths for testing
        batch_size = 2 
        dummy_video_paths_for_test = []
        try:
            from video_modality import create_dummy_video
            for i in range(batch_size):
                p = f"test_video_main_model_{i}.mp4"
                create_dummy_video(path=p, num_frames_total=10, fps=10, width=224, height=224)
                if os.path.exists(p):
                    dummy_video_paths_for_test.append(p)
            if not dummy_video_paths_for_test and batch_size > 0:
                 print("ERROR: No dummy videos were created for the test. Aborting test run.")                 
                 exit()
        except ImportError:
            print("Skipping dummy video creation as `create_dummy_video` not found.")
            if os.path.exists("test_video.mp4"): 
                 dummy_video_paths_for_test.append("test_video.mp4")
            if not dummy_video_paths_for_test:
                 print("ERROR: No video paths available for testing. Aborting.")
                 exit()
        
        if not dummy_video_paths_for_test:
            print("No video paths available for testing. The __main__ block might not run correctly.")
        else:
            # --- Test Stage 1 --- 
            print(f"\n--- Simulating Stage 1 Forward Pass with {len(dummy_video_paths_for_test)} videos ---")
            total_loss, video_loss, audio_loss = t3_model.forward_stage1(
                dummy_video_paths_for_test, device, train_video_mae=True, train_audio_mae=True
            )
            print(f"Stage 1 Total MAE Loss: {total_loss.item()}")
            if video_loss is not None: print(f"  Stage 1 Video MAE Loss: {video_loss}")
            if audio_loss is not None: print(f"  Stage 1 Audio MAE Loss: {audio_loss}")
            
            # --- Test Stage 2 --- 
            print(f"\n--- Simulating Stage 2 Forward Pass with {len(dummy_video_paths_for_test)} videos ---")
            contrastive_loss, b_proc_s2 = t3_model.forward_stage2(
                 dummy_video_paths_for_test, device=device
            )
            print(f"Stage 2 Contrastive Loss: {contrastive_loss.item()} (processed {b_proc_s2} samples)")
            
            # --- Test Fine-tuning --- 
            print(f"\n--- Simulating AV Fused Fine-tuning Forward Pass with {len(dummy_video_paths_for_test)} videos ---")
            num_classes_test = sound_clf_head_params_for_test['num_classes']
            # Ensure dummy_labels matches the number of successfully processed items if that changes
            # For this test, assume all paths in dummy_video_paths_for_test are processed successfully by both encoders
            effective_batch_size_for_labels = len(dummy_video_paths_for_test) 
            dummy_labels = torch.randint(0, num_classes_test, (effective_batch_size_for_labels,), device=device)
            print(f"Using dummy labels: {dummy_labels.tolist()}")
            
            finetune_loss, logits, processed_ft_bsz = t3_model.forward_finetune(
                dummy_video_paths_for_test, dummy_labels, device=device
            )
            print(f"Finetune AV Classification Loss: {finetune_loss.item()} (processed {processed_ft_bsz} samples)")
            print(f"Output Logits Shape: {logits.shape}") 
            if logits.numel() > 0 and processed_ft_bsz > 0:
                preds = torch.argmax(logits, dim=1)
                print(f"Predicted classes: {preds.tolist()}")

    except Exception as e:
        print(f"ERROR during T3_AV_Model __main__ example: {e}")
        import traceback
        traceback.print_exc()
        
    print("\nFinished T3_AV_Model tests.") 