import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.distributed as dist

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

# Helper for DDP AllGather (ensure this is defined in your file)
class AllGatherFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor_list_dummy, tensor_to_gather): # tensor_list_dummy is not used for backward/forward, just to match signature if needed
        if not dist.is_available() or not dist.is_initialized():
            return tensor_to_gather.unsqueeze(0) # Simulate list of one tensor if not DDP

        world_size = dist.get_world_size()
        # Ensure tensor_to_gather is on the correct device, though it should be already
        # tensor_to_gather = tensor_to_gather.to(f'cuda:{dist.get_rank()}') # Usually already on correct device

        # Create a list of tensors for gathering, even if some are empty
        # All tensors in this list must be on the GPU for nccl backend
        gathered_tensors_list = [
            torch.empty_like(tensor_to_gather) for _ in range(world_size)
        ]
        dist.all_gather(gathered_tensors_list, tensor_to_gather)
        return torch.cat(gathered_tensors_list, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return None, grad_output # No DDP or single GPU

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Calculate the size of the tensor for this rank based on how grad_output was formed
        # This assumes grad_output corresponds to the concatenated tensor from forward
        # and that the original tensors had the same number of columns.
        # The split must align with how the forward tensors were cat'd.
        # If forward cat'd tensors [B0,D], [B1,D], ..., grad_output is [sum(Bi), D]
        # We need to find which part of grad_output corresponds to this rank's original tensor.
        # This requires knowing the original B_local for each rank, which is not directly available in backward.
        # A common simplification if all B_local are assumed equal for backward distribution:
        # grad_input_local = grad_output.chunk(world_size, dim=0)[rank]
        # However, if B_local varies, this is incorrect.
        # For a robust backward with varying B_local, one might need to save B_locals in ctx
        # or use a simpler gather like `all_reduce` for the loss itself if possible.

        # Given the AllGatherFunc structure, the backward is tricky if B_local varies.
        # A common pattern for contrastive loss is that the loss is computed per GPU
        # using local queries and global keys, and then losses are averaged.
        # The gradient of `z_local` (query) is direct.
        # The gradient of `z_all_keys` needs to be scattered.
        # The provided AllGatherFunc's backward is a simplification.
        # For a fully correct backward with varying B_local, it's more complex.
        # However, many libraries use this simplified backward or rely on PyTorch's DDP to handle it.
        # Let's stick to the common simplified version for now:
        
        # Determine split sizes. This is the hard part if B_local varies.
        # For simplicity, if we assume the grad_output can be chunked equally (which is an approximation if B_local varies significantly)
        # grad_input_local = grad_output.chunk(world_size, dim=0)[rank]
        
        # A more correct approach for backward with varying sizes is to ensure that
        # the gradients are only passed back to the parts of the input that generated them.
        # If the loss is computed using local_query vs all_keys, the gradient w.r.t local_query is direct.
        # The gradient w.r.t all_keys needs to be handled.
        # `torch.distributed.nn.functional.all_gather` (newer PyTorch) might handle this better.

        # Sticking to the provided AllGatherFunc's backward:
        # This backward assumes that the contribution to grad_output can be evenly split,
        # or that the part relevant to this rank can be extracted by its rank index.
        # This is an approximation if local batch sizes varied.
        
        # Let's assume the grad_output corresponds to the concatenated tensor.
        # The input `tensor_to_gather` was (B_local, D).
        # `grad_output` is (B_global, D).
        # We need to extract the part of `grad_output` that corresponds to this rank's `tensor_to_gather`.
        # This requires knowing the B_local of *all* ranks to find the correct slice.
        # This information is not typically passed to backward.
        # This highlights a common challenge with custom AllGather for varying batch sizes.

        # Fallback to a simpler gradient distribution, acknowledging its approximation:
        # If we can assume that the optimizer step will handle scaling,
        # we can try to pass back the relevant slice.
        # However, without knowing all B_locals, it's hard.
        # The most common use of such an AllGather is when the loss is computed,
        # and then `loss.backward()` handles distributing gradients through the DDP graph.
        # The `AllGatherFunc.apply(z_local)` makes `z_local` a leaf in this part of the graph for other GPUs.
        
        # Simplest backward:
        # grad_input_local = grad_output.clone() # Each gets the full grad, DDP averages it. (Often done)
        # Or, if we assume grad_output is for the concatenated tensor, and we need grad for the local input tensor:
        # This is where it gets tricky. The provided backward:
        grad_input_local = grad_output.chunk(world_size, dim=0)[rank] # This assumes B_local is same for all ranks
        return None, grad_input_local # Grad for tensor_list_dummy is None

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
        self.contrastive_embed_dim = contrastive_embed_dim

        # 1. Instantiate Modality Encoders
        self.video_encoder = VideoModality(**video_modality_params)
        self.audio_encoder = AudioModality(**audio_modality_params)
        print("Actual VideoModality and AudioModality instantiated.")

        # Ensure both encoders use the same hidden dimension from their ViTs for shared backbone compatibility
        assert self.video_encoder.vit_hidden_size == self.audio_encoder.vit_hidden_size, \
            "Video and Audio ViT hidden sizes must match for shared backbone input."
        self.modal_embed_dim = self.video_encoder.vit_hidden_size # This will be embed_dim for shared_backbone

        # The actual sequence length passed to shared_backbone.forward depends on the stage:
        # Stage 1: num_unmasked + 1
        # Stage 2/Finetune: total_patches + 1
        # Therefore, PositionalEncoding max_len must accommodate the largest possible full sequence.
        max_total_patches_video = self.video_encoder.mae_total_patches
        max_total_patches_audio = self.audio_encoder.mae_num_patches
        max_full_seq_len_needed = max(max_total_patches_video, max_total_patches_audio) + 1
        
        _shared_backbone_params = {
            'embed_dim': self.modal_embed_dim,
            'max_seq_len': max(max_full_seq_len_needed, 256), # Use max total patches + CLS, ensure min length
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
    def _get_empty_features(self, device):
        # Helper to create an empty tensor with the correct shape and dtype for features
        # Used when a rank processes 0 local samples but needs to participate in all_gather
        example_param = next(self.parameters()) # Get dtype and device context
        return torch.empty((0, self.contrastive_embed_dim), device=device, dtype=example_param.dtype)

    def forward_stage2(self, video_paths_batch, device):
        # 1. Get local features (as in your current working version)
        video_features_raw = self.video_encoder(video_paths_batch, device=device) # (B_local_v, N_v+1, D_modal)
        audio_features_raw = self.audio_encoder(video_paths_batch, device=device) # (B_local_a, N_a+1, D_modal)

        B_local_v = video_features_raw.shape[0]
        B_local_a = audio_features_raw.shape[0]
        
        current_rank_for_log = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        if B_local_v != B_local_a:
            print(f"Rank {current_rank_for_log}: Local batch size mismatch in Stage 2. Video: {B_local_v}, Audio: {B_local_a}. Using 0 for this rank.")
            B_local = 0
        else:
            B_local = B_local_v # or B_local_a

        if B_local == 0:
            # This rank has no valid local data, prepare empty tensors for all_gather
            z_a = self._get_empty_features(device)
            z_v = self._get_empty_features(device)
        else:
            # Proceed with backbone and projection for this rank's valid local batch
            bb_out_v = self.shared_backbone(video_features_raw)
            bb_out_a = self.shared_backbone(audio_features_raw)
            z_v_local_proj = self.video_projection_head(bb_out_v, use_cls_token=True)
            z_a_local_proj = self.audio_projection_head(bb_out_a, use_cls_token=True)
            
            z_v = F.normalize(z_v_local_proj, dim=1, p=2)
            z_a = F.normalize(z_a_local_proj, dim=1, p=2)

        # All ranks MUST call all_gather
        is_ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        
        if is_ddp:
            z_a_all = AllGatherFunc.apply(None, z_a) # z_a can be (0,D)
            z_v_all = AllGatherFunc.apply(None, z_v) # z_v can be (0,D)
            
            # Gather local batch sizes from all ranks
            local_b_tensor = torch.tensor([B_local], device=device, dtype=torch.long)
            world_b_list = [torch.zeros_like(local_b_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(world_b_list, local_b_tensor)
            world_batch_sizes = [b.item() for b in world_b_list] # List of B_local for each rank [B0, B1,...]
            current_rank = dist.get_rank()
        else:
            z_a_all = z_a
            z_v_all = z_v
            world_batch_sizes = [B_local]
            current_rank = 0
            
        total_contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Compute loss only if the current rank has actual local data (B_local > 0)
        # AND if there's any data globally to compare against.
        if B_local > 0 and z_v_all.shape[0] > 0: # Check z_v_all for audio query vs video keys
            loss_a_v = self._compute_infonce_loss_ddp(z_a, z_v_all, self.temperature, current_rank, world_batch_sizes)
            total_contrastive_loss = total_contrastive_loss + loss_a_v
        
        if B_local > 0 and z_a_all.shape[0] > 0: # Check z_a_all for video query vs audio keys
            loss_v_a = self._compute_infonce_loss_ddp(z_v, z_a_all, self.temperature, current_rank, world_batch_sizes)
            total_contrastive_loss = total_contrastive_loss + loss_v_a
        
        if B_local > 0 and (z_v_all.shape[0] > 0 or z_a_all.shape[0] > 0) : # Avoid division by zero if both losses were computed
             total_contrastive_loss = total_contrastive_loss / 2.0 # Average if both components were added

        # The `batch_size_processed` returned is the local batch size for this rank.
        # The training loop in train_stage2.py will sum these up for global average loss calculation.
        return total_contrastive_loss, B_local

    def _compute_infonce_loss_ddp(self, z_query_local, z_keys_all, temperature, current_rank, world_batch_sizes):
        # z_query_local: (B_local_current_rank, D) - features from the current GPU.
        # z_keys_all: (B_global, D) - features gathered from all GPUs.
        # temperature: scalar.
        # current_rank: current GPU rank.
        # world_batch_sizes: list of B_local for each rank [B0, B1, B2, ...]
        
        B_local_current_rank = z_query_local.shape[0]
        
        # This function should only be called if B_local_current_rank > 0
        if B_local_current_rank == 0:
            # This case should ideally be handled before calling this function.
            return torch.tensor(0.0, device=z_query_local.device, dtype=z_query_local.dtype, requires_grad=True)

        if z_keys_all.shape[0] == 0: # No keys gathered from any rank
             return torch.tensor(0.0, device=z_query_local.device, dtype=z_query_local.dtype, requires_grad=True)

        # Logits: (B_local_current_rank, B_global)
        logits = torch.mm(z_query_local, z_keys_all.t()) / temperature
        
        # Calculate the starting index (offset) of the current rank's positive keys within z_keys_all
        # This offset is the sum of batch sizes of all preceding ranks
        offset = sum(world_batch_sizes[:current_rank])
        
        # Labels point to the positive keys for each local query
        # These are indices into z_keys_all
        labels = torch.arange(B_local_current_rank, device=logits.device, dtype=torch.long) + offset
        
        loss = F.cross_entropy(logits, labels)
        return loss

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