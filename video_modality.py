import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
import cv2
import numpy as np
from PIL import Image
import os

def create_dummy_video(path="dummy_video.mp4", num_frames_total=50, fps=10, width=224, height=224):
    """Creates a dummy MP4 video file for testing."""
    if os.path.exists(path):
        print(f"Dummy video {path} already exists. Skipping creation.")
        return
    print(f"Creating dummy video at {path}...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for path {path}")
        return
        
    for i in range(num_frames_total):
        # Create a frame with a color gradient that changes over time
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        r_val = (i * 5) % 256
        g_val = (i * 3) % 256
        b_val = (i * 2) % 256
        frame[:, :, 0] = b_val  # Blue channel
        frame[:, :, 1] = g_val  # Green channel
        frame[:, :, 2] = r_val  # Red channel
        cv2.putText(frame, f"Frame {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    print(f"Dummy video {path} created successfully with {num_frames_total} frames.")

class VideoModality(nn.Module):
    """
    Processes a video file to extract a sequence of feature embeddings.
    1. Extracts frames.
    2. Uses a pre-trained ViT to get per-frame features.
    3. Applies a linear projection.
    4. Adds positional embeddings.
    """
    def __init__(self, num_frames_to_sample=16, pretrained_vit_name='google/vit-base-patch16-224-in21k', output_embed_dim=768):
        super().__init__()
        self.num_frames_to_sample = num_frames_to_sample
        self.output_embed_dim = output_embed_dim # For original forward method (contrastive)
        self.pretrained_vit_name = pretrained_vit_name # Store for MAE patch size access

        print(f"Initializing VideoModality with {pretrained_vit_name}...")
        try:
            self.image_processor = ViTImageProcessor.from_pretrained(pretrained_vit_name)
            self.vit_model = ViTModel.from_pretrained(pretrained_vit_name)
        except Exception as e:
            print(f"Error loading ViT model/processor: {e}")
            print("Please ensure you have an internet connection and the `transformers` library is installed correctly.")
            raise

        self.vit_hidden_size = self.vit_model.config.hidden_size # Also needed for MAE patch embeddings
        
        # These are for the original 'forward' method (contrastive learning features)
        self.linear_projection = nn.Linear(self.vit_hidden_size, output_embed_dim)
        self.positional_embeddings = nn.Parameter(torch.randn(1, num_frames_to_sample, output_embed_dim))
        
        # For MAE: patch size and derived dimensions
        self.mae_patch_size = self.vit_model.config.patch_size
        self.mae_target_height = self.image_processor.size['height']
        self.mae_target_width = self.image_processor.size['width']
        self.mae_num_patches_per_frame = (self.mae_target_height // self.mae_patch_size) * \
                                         (self.mae_target_width // self.mae_patch_size)
        self.mae_total_patches = self.num_frames_to_sample * self.mae_num_patches_per_frame
        # Dimension of a single flattened patch: C * H * W (e.g., 3 * 16 * 16)
        self.mae_patch_pixel_dim = 3 * self.mae_patch_size * self.mae_patch_size 

        print("VideoModality initialized.")

    def _extract_frames(self, video_path: str) -> list[Image.Image]:
        """
        Extracts `self.num_frames_to_sample` frames from the video, converts them to PIL Images.
        Handles videos with fewer frames by padding with the last frame.
        Returns black frames if video cannot be read or has no frames.
        """
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Cannot open video file: {video_path}")
                # Return black frames if video can't be opened
                default_size = (self.image_processor.size['height'], self.image_processor.size['width'])
                return [Image.new('RGB', default_size, (0, 0, 0))] * self.num_frames_to_sample

            total_actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_actual_frames == 0:
                print(f"Warning: Video file {video_path} has 0 frames.")
                cap.release()
                default_size = (self.image_processor.size['height'], self.image_processor.size['width'])
                return [Image.new('RGB', default_size, (0, 0, 0))] * self.num_frames_to_sample

            frame_indices_to_grab = np.linspace(0, total_actual_frames - 1, self.num_frames_to_sample, dtype=int)
            
            for i, frame_idx in enumerate(frame_indices_to_grab):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame_bgr = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
                else:
                    # If reading fails (e.g., end of video or corrupted frame), use last good frame or black frame
                    if frames: # Pad with last good frame
                        frames.append(frames[-1])
                    else: # No good frames yet, use a black frame
                        default_size = (self.image_processor.size['height'], self.image_processor.size['width'])
                        frames.append(Image.new('RGB', default_size, (0,0,0)))
            cap.release()

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Fallback to black frames in case of any other error
            default_size = (self.image_processor.size['height'], self.image_processor.size['width'])
            return [Image.new('RGB', default_size, (0, 0, 0))] * self.num_frames_to_sample
        
        # Ensure we always return the exact number of frames expected
        if len(frames) < self.num_frames_to_sample and frames:
             frames.extend([frames[-1]] * (self.num_frames_to_sample - len(frames)))
        elif not frames: # Should be covered by earlier checks, but as a safeguard
            default_size = (self.image_processor.size['height'], self.image_processor.size['width'])
            return [Image.new('RGB', default_size, (0, 0, 0))] * self.num_frames_to_sample
            
        return frames[:self.num_frames_to_sample]

    def _frames_to_patches(self, pil_frames: list[Image.Image], device: torch.device) -> torch.Tensor:
        """
        Converts a list of PIL frames to a batch of flattened patches (C*H*W) for MAE target.
        Assumes frames are already resized to self.mae_target_height, self.mae_target_width.
        Args:
            pil_frames: List of PIL.Image objects.
            device: Torch device.
        Returns:
            A tensor of shape (num_frames, num_patches_per_frame, C*patch_H*patch_W)
        """
        all_frame_patches_list = []
        for pil_frame in pil_frames:
            # Ensure frame is correct size (should be by _extract_frames if it uses image_processor.size)
            img_tensor = self.image_processor.preprocess(pil_frame, return_tensors='pt')['pixel_values'][0].to(device) # (C, H, W)
            
            # Unfold to get patches: (C, H, W) -> (C, num_patches_h, patch_size_h, num_patches_w, patch_size_w)
            # Then permute and view: (C, num_patches_h, num_patches_w, patch_size_h, patch_size_w)
            # -> (num_patches_h * num_patches_w, C, patch_size_h, patch_size_w)
            # -> (num_patches_per_frame, C * patch_size_h * patch_size_w)
            
            C, H, W = img_tensor.shape
            ps = self.mae_patch_size
            num_patches_h = H // ps
            num_patches_w = W // ps

            # (C, num_patches_h, ps, num_patches_w, ps)
            patches = img_tensor.unfold(1, ps, ps).unfold(2, ps, ps) 
            # (C, num_patches_h, num_patches_w, ps, ps)
            patches = patches.permute(0, 1, 3, 2, 4).contiguous() 
            # (num_patches_h, num_patches_w, C, ps, ps)
            patches = patches.view(C, num_patches_h, num_patches_w, ps, ps).permute(1,2,0,3,4).contiguous()
            # (num_patches_per_frame, C*ps*ps)
            patches = patches.view(num_patches_h * num_patches_w, C * ps * ps)
            all_frame_patches_list.append(patches)

        return torch.stack(all_frame_patches_list) # (num_frames, num_patches_per_frame, C*ps*ps)

    def get_patches_for_mae(self, video_path: str, device: torch.device):
        """
        Processes a single video file for MAE.
        Returns:
            all_patch_embeddings: (1, total_video_patches, vit_hidden_size) - Embeddings from ViT's last_hidden_state.
            original_pixel_patches: (1, total_video_patches, C*patch_H*patch_W) - Flattened original pixel values of patches.
            total_video_patches: Integer, total number of patches (num_frames * num_patches_per_frame).
        """
        pil_frames = self._extract_frames(video_path) # List of num_frames_to_sample PIL Images

        # 1. Get original pixel patches for MAE target
        # Each PIL frame is (H, W, C) after conversion. image_processor.preprocess makes it (C,H,W) tensor.
        # Ensure frames are resized to ViT's expected input size before patching.
        # The _extract_frames should ideally use image_processor.size for default_size
        # and ViTImageProcessor.preprocess will resize during its call.
        # For _frames_to_patches, we need frames at the ViT input resolution.
        # A robust way is to first process them by image_processor, then unpatch.
        
        # Process all frames at once for ViT input
        try:
            inputs = self.image_processor(images=pil_frames, return_tensors="pt").to(device)
            pixel_values_for_vit = inputs['pixel_values'] # (num_frames_to_sample, C, H, W)
        except Exception as e:
            print(f"Error during image_processor in get_patches_for_mae: {e}")
            # Return dummy tensors on error
            return (torch.zeros(1, self.mae_total_patches, self.vit_hidden_size, device=device),
                    torch.zeros(1, self.mae_total_patches, self.mae_patch_pixel_dim, device=device),
                    self.mae_total_patches)

        # Manually create patches from pixel_values_for_vit (which are already processed)
        # for the MAE reconstruction target.
        # pixel_values_for_vit is (num_frames, C, H_vit, W_vit)
        # We need to unpatch this to (num_frames, num_patches_per_frame, C*ps*ps)
        # then reshape to (1, num_frames * num_patches_per_frame, C*ps*ps)
        
        C, H_vit, W_vit = pixel_values_for_vit.shape[1], pixel_values_for_vit.shape[2], pixel_values_for_vit.shape[3]
        ps = self.mae_patch_size
        num_p_h = H_vit // ps
        num_p_w = W_vit // ps

        target_patches_list = []
        for frame_idx in range(self.num_frames_to_sample):
            frame_tensor = pixel_values_for_vit[frame_idx] # (C, H_vit, W_vit)
            # (C, num_p_h, ps, num_p_w, ps)
            patches_unfold = frame_tensor.unfold(1, ps, ps).unfold(2, ps, ps)
            # (C, num_p_h, num_p_w, ps, ps)
            patches_permute = patches_unfold.permute(0, 1, 3, 2, 4).contiguous()
            # (num_p_h, num_p_w, C, ps, ps)
            patches_view = patches_permute.view(C, num_p_h, num_p_w, ps, ps).permute(1,2,0,3,4).contiguous()
            # (num_p_h * num_p_w, C * ps * ps)
            flattened_patches = patches_view.view(-1, C * ps * ps)
            target_patches_list.append(flattened_patches)
        
        original_pixel_patches_all_frames = torch.cat(target_patches_list, dim=0) # (total_video_patches, C*ps*ps)
        original_pixel_patches_all_frames = original_pixel_patches_all_frames.unsqueeze(0) # (1, total_video_patches, C*ps*ps)


        # 2. Get patch embeddings from ViT (last_hidden_state)
        with torch.no_grad(): 
            vit_outputs = self.vit_model(pixel_values=pixel_values_for_vit)
        
        patch_embeddings_from_vit = vit_outputs.last_hidden_state[:, 1:, :] # Exclude CLS token
        
        B_frames, N_patches_p_frame, D_hidden = patch_embeddings_from_vit.shape
        all_patch_embeddings = patch_embeddings_from_vit.reshape(1, B_frames * N_patches_p_frame, D_hidden)

        assert all_patch_embeddings.shape[1] == self.mae_total_patches, \
            f"Mismatch in total patches for embeddings: expected {self.mae_total_patches}, got {all_patch_embeddings.shape[1]}"
        assert original_pixel_patches_all_frames.shape[1] == self.mae_total_patches, \
            f"Mismatch in total patches for original pixels: expected {self.mae_total_patches}, got {original_pixel_patches_all_frames.shape[1]}"

        return all_patch_embeddings, original_pixel_patches_all_frames, self.mae_total_patches

    def forward(self, video_paths_batch: list[str], device: torch.device) -> torch.Tensor:
        """
        Processes a BATCH of video files for contrastive learning (Stage 2).
        Extracts frames, processes with ViT, and returns patch embeddings suitable
        for the SharedTransformerBackbone.
        Args:
            video_paths_batch: List of paths to video files.
            device: Torch device.
        Returns:
            A tensor of patch embeddings shape (batch_size, total_video_patches, vit_hidden_size)
            Returns zeros tensor if batch fails.
        """
        batch_size = len(video_paths_batch)
        batch_embeddings_list = []
        processed_successfully = 0

        for video_path in video_paths_batch:
            pil_frames = self._extract_frames(video_path)
            try:
                inputs = self.image_processor(images=pil_frames, return_tensors="pt").to(device)
                pixel_values_for_vit = inputs['pixel_values'] # (num_frames, C, H, W)
                
                # Pass frames through ViT (treating frames as batch dim for ViT)
                # NO GRADIENT needed if encoders are frozen in Stage 2, but keep for flexibility
                # Consider adding a flag or checking self.training if freezing is desired here.
                # with torch.no_grad() if not self.training else nullcontext(): 
                vit_outputs = self.vit_model(pixel_values=pixel_values_for_vit)
                
                # Get patch embeddings (excluding CLS token from ViT internal embeddings)
                patch_embeddings_from_vit = vit_outputs.last_hidden_state[:, 1:, :] # (num_frames, num_patches_p_frame, hidden)
                
                # Reshape to (1, total_video_patches, hidden_size) for this single video
                B_frames, N_patches_p_frame, D_hidden = patch_embeddings_from_vit.shape
                single_video_embeddings = patch_embeddings_from_vit.reshape(1, B_frames * N_patches_p_frame, D_hidden)
                
                # Basic check
                if single_video_embeddings.shape[1] != self.mae_total_patches:
                     print(f"Warning: Patch count mismatch for {video_path}. Expected {self.mae_total_patches}, got {single_video_embeddings.shape[1]}. Skipping.")
                     continue # Skip this video

                batch_embeddings_list.append(single_video_embeddings)
                processed_successfully += 1

            except Exception as e:
                print(f"Error processing video {video_path} in forward pass: {e}. Skipping.")
                # Optionally append zeros or handle differently
                continue 
        
        if processed_successfully == 0 and batch_size > 0:
             print(f"Error: All videos in the batch failed processing in VideoModality forward.")
             # Return zeros tensor matching expected output shape for B=batch_size
             return torch.zeros(batch_size, self.mae_total_patches, self.vit_hidden_size, device=device)
        elif not batch_embeddings_list: # If batch_size was 0 or all failed
             return torch.zeros(0, self.mae_total_patches, self.vit_hidden_size, device=device)

        # Concatenate the list of (1, N, D) tensors into (B_processed, N, D)
        final_batch_embeddings = torch.cat(batch_embeddings_list, dim=0)
        
        # If some videos failed, the batch size might be smaller than input batch_size.
        # The calling function (T3_AV_Model.forward_stage2) needs to handle this, maybe by aligning
        # video and audio batches based on successfully processed items. Simpler for now is to return
        # the successfully processed ones.
        return final_batch_embeddings 

if __name__ == '__main__':
    print("Running VideoModality example...")
    
    dummy_video_path = "test_video.mp4"

    if not os.path.exists(dummy_video_path):
        print(f"Failed to create or find dummy video: {dummy_video_path}. Exiting example.")
    else:
        # Instantiate the model
        # Using a smaller ViT for faster local testing if needed, e.g. 'google/vit-small-patch16-224'
        # For now, stick to the specified 'google/vit-base-patch16-224-in21k'
        try:
            video_processor = VideoModality(num_frames_to_sample=16, output_embed_dim=512)
            
            # Set to evaluation mode if not training
            video_processor.eval() 
            
            print(f"Processing video: {dummy_video_path}")
            # Process the dummy video
            # Ensure the model is on a device, e.g. CPU for this example
            # If CUDA is available and you want to use it:
            # if torch.cuda.is_available():
            #    video_processor = video_processor.to('cuda')
            
            video_features = video_processor(dummy_video_path)
            
            print(f"Output feature shape: {video_features.shape}") # Expected: (1, 16, 512)
            print("First feature vector (first 5 elements):", video_features[0, 0, :5].tolist())

            # Test with a non-existent video
            print("\nProcessing non-existent video...")
            non_existent_video_features = video_processor("non_existent_video.mp4")
            print(f"Output for non-existent video (shape): {non_existent_video_features.shape}")
            # Expected: (1, 16, 512) with zeros or based on black frames

            # --- Test get_patches_for_mae ---
            print("\n--- Testing get_patches_for_mae ---")
            if os.path.exists(dummy_video_path):
                # Re-instantiate or use existing video_processor if already on device
                # For simplicity, let's assume video_processor is available and on CPU for this test part
                video_processor.eval() # Ensure it's in eval mode
                
                # Get device from the model (or set explicitly)
                current_device = next(video_processor.parameters()).device
                print(f"Using device for MAE test: {current_device}")

                try:
                    embeddings, original_patches, total_patches = video_processor.get_patches_for_mae(dummy_video_path, device=current_device)
                    print(f"  MAE Patch Embeddings Shape: {embeddings.shape}")
                    # Expected: (1, num_frames_to_sample * num_patches_per_frame, vit_hidden_size)
                    # e.g., (1, 16 * (224/16)^2, 768) = (1, 16 * 196, 768) = (1, 3136, 768)
                    print(f"  MAE Original Pixel Patches Shape: {original_patches.shape}")
                    # Expected: (1, num_frames_to_sample * num_patches_per_frame, 3 * patch_size * patch_size)
                    # e.g., (1, 3136, 3*16*16=768)
                    print(f"  MAE Total Patches: {total_patches}")
                    # Expected: 16 * 196 = 3136
                    
                    # Verify derived MAE dimensions
                    print(f"  Config: mae_patch_size={video_processor.mae_patch_size}")
                    print(f"  Config: mae_num_patches_per_frame={video_processor.mae_num_patches_per_frame}")
                    print(f"  Config: mae_total_patches={video_processor.mae_total_patches}")
                    print(f"  Config: mae_patch_pixel_dim={video_processor.mae_patch_pixel_dim}")


                except Exception as e_mae:
                    print(f"Error during get_patches_for_mae test: {e_mae}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Skipping get_patches_for_mae test as dummy video not found.")
            # --- End Test get_patches_for_mae ---

            # --- Test Batch Forward --- 
            print("\n--- Testing Batch Forward (Stage 2 features) ---")
            test_batch_paths = []
            if os.path.exists(dummy_video_path): test_batch_paths.append(dummy_video_path)
            if os.path.exists("test_video_main_model_0.mp4"): test_batch_paths.append("test_video_main_model_0.mp4")
            if os.path.exists("test_video_main_model_1.mp4"): test_batch_paths.append("test_video_main_model_1.mp4")

            if len(test_batch_paths) >= 1:
                print(f"Processing batch of {len(test_batch_paths)} videos...")
                try:
                     batch_features = video_processor(test_batch_paths, device=current_device)
                     print(f"  Batch Forward Output Shape: {batch_features.shape}")
                     # Expected: (len(test_batch_paths), mae_total_patches, vit_hidden_size)
                     # e.g., (B, 3136, 768)
                except Exception as e_fwd:
                    print(f"Error during batch forward test: {e_fwd}")
                    import traceback
                    traceback.print_exc()
            else:
                 print("Skipping batch forward test as no suitable test videos found.")
            # --- End Test Batch Forward ---

        except ImportError as e:
            print(f"Import error: {e}. Please ensure all dependencies (torch, transformers, opencv-python, numpy, Pillow) are installed.")
        except Exception as e:
            print(f"An error occurred during the example run: {e}")
            import traceback
            traceback.print_exc()

    print("\nVideoModality example finished.") 