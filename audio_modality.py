import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from transformers import ViTModel, ViTImageProcessor
from moviepy import *
# from moviepy.editor import AudioFileClip
import numpy as np
from PIL import Image
import os
import io

# Try to import the dummy video creator from video_modality
# This is for convenient testing if both files are in the same directory
try:
    from video_modality import create_dummy_video
except ImportError:
    print("Warning: `create_dummy_video` from `video_modality.py` not found. Testing may require a manual dummy video.")
    def create_dummy_video(path="dummy_video.mp4", **kwargs):
        if not os.path.exists(path):
            print(f"Placeholder: Dummy video {path} would be created here if video_modality.py was available.")
            # Create a tiny placeholder if it absolutely doesn't exist, so moviepy doesn't fail hard on path not found
            # This won't have valid video/audio content for actual processing by video_modality though.
            # For audio_modality, moviepy might still extract silence or error if the file is malformed.
            with open(path, 'w') as f:
                f.write("dummy")
        else:
            print(f"Dummy video {path} already exists (or is a placeholder).")


class AudioModality(nn.Module):
    """
    Processes audio from a video file to extract a sequence of feature embeddings.
    1. Extracts audio from video.
    2. Converts audio to Mel spectrogram.
    3. Treats spectrogram as an image and uses ViT to get patch embeddings.
    4. Applies a linear projection and adds positional embeddings.
    """
    def __init__(self,
                 sample_rate=16000,
                 n_mels=128,
                 n_fft=1024, # For 16kHz, 1024 is ~64ms window
                 hop_length=512, # For 16kHz, 512 is ~32ms hop
                 target_spectrogram_height=224, # ViT standard input size
                 target_spectrogram_width=224,  # ViT standard input size
                 pretrained_vit_name='google/vit-base-patch16-224-in21k',
                 output_embed_dim=768, # Should match video modality if concatenated later
                 audio_duration_secs=10): # Expected audio duration from VGGSound
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_spectrogram_height = target_spectrogram_height
        self.target_spectrogram_width = target_spectrogram_width
        self.audio_duration_secs = audio_duration_secs
        self.output_embed_dim = output_embed_dim
        self.pretrained_vit_name = pretrained_vit_name

        print(f"Initializing AudioModality with {pretrained_vit_name}...")
        try:
            self.image_processor = ViTImageProcessor.from_pretrained(pretrained_vit_name)
            self.vit_model = ViTModel.from_pretrained(pretrained_vit_name)
        except Exception as e:
            print(f"Error loading ViT model/processor for audio: {e}")
            raise

        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        self.amplitude_to_db_transform = T.AmplitudeToDB()

        self.vit_hidden_size = self.vit_model.config.hidden_size
        self.mae_patch_size = self.vit_model.config.patch_size
        
        # Num patches for the spectrogram image fed to ViT (after resizing by image_processor)
        self.mae_num_patches = (self.target_spectrogram_height // self.mae_patch_size) * \
                               (self.target_spectrogram_width // self.mae_patch_size)
        
        # For MAE target, original patch is 1 channel (spectrogram value)
        # If ViT image_processor converts to 3 channels, target should still be 1 channel
        self.mae_patch_spectrogram_dim = 1 * self.mae_patch_size * self.mae_patch_size

        # Components for original 'forward' method (contrastive learning)
        # The number of patches for original forward method's positional_embeddings should be mae_num_patches
        self.linear_projection = nn.Linear(self.vit_hidden_size, output_embed_dim)
        self.positional_embeddings = nn.Parameter(torch.randn(1, self.mae_num_patches, output_embed_dim))
        print("AudioModality initialized.")

    def _extract_audio_waveform(self, video_path: str) -> tuple[torch.Tensor | None, int | None]:
        """Extracts audio waveform and sample rate from video file using torchaudio primarily."""
        try:
            if not os.path.exists(video_path):
                print(f"Error: Video file not found at {video_path}")
                return None, None

            # Try loading directly with torchaudio
            waveform, sr = torchaudio.load(video_path)

            # Resample if necessary
            if sr != self.sample_rate:
                resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
                sr = self.sample_rate # Update sample rate after resampling

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Ensure target duration
            num_target_samples = int(self.sample_rate * self.audio_duration_secs) # Ensure int
            current_samples = waveform.shape[1]

            if current_samples < num_target_samples: # Pad if shorter
                padding = num_target_samples - current_samples
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif current_samples > num_target_samples: # Truncate if longer
                waveform = waveform[:, :num_target_samples]
            
            return waveform, sr # sr should be self.sample_rate now

        except Exception as e_torchaudio:
            print(f"Torchaudio direct load failed for {video_path}: {e_torchaudio}. Trying moviepy as fallback (if implemented).")
            # Fallback to moviepy (original logic) if torchaudio fails - or remove moviepy path if it's problematic
            # For now, let's keep the moviepy path as a potential fallback, but ensure it also handles errors well.
            try:
                audio_clip = AudioFileClip(video_path)
                # If moviepy is kept, its internal writing to BytesIO and then loading needs to be robust.
                # Given the error, it might be safer to simplify or remove this path if torchaudio is generally reliable.
                # For now, the original moviepy path after this point is problematic based on the error.
                # Let's prioritize the torchaudio path working.
                # If we must use moviepy, ensure AudioFileClip() itself is not the source of the path-like error.
                
                # Re-evaluating the moviepy path given the error:
                # The error "expected str, bytes or os.PathLike object, not BytesIO"
                # is most likely from an internal ffmpeg call triggered by moviepy
                # that doesn't correctly handle a BytesIO object passed internally
                # when it expected a file path string after the initial AudioFileClip(video_path).
                #
                # The most robust solution is to rely on torchaudio's direct file handling
                # and ffmpeg backend which is generally more straightforward for this task.

                # If torchaudio direct load fails, we might assume the file is problematic for direct audio loading.
                print(f"Moviepy fallback for {video_path} also skipped due to previous issues with its path handling.")
                return None, None # Skip if torchaudio direct load fails.

            except Exception as e_moviepy:
                 print(f"Moviepy fallback also failed for {video_path}: {e_moviepy}")
                 return None, None

    def _preprocess_spectrogram(self, mel_spec: torch.Tensor) -> Image.Image:
        """Converts spectrogram tensor to a PIL Image suitable for ViTImageProcessor."""
        # Normalize for image conversion (e.g., to 0-255 range)
        mel_spec_db = self.amplitude_to_db_transform(mel_spec)
        
        # Normalize to [0, 1] for image conversion
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
        
        # Convert to PIL Image
        # Permute to (H, W, C) for PIL if starting from (C, H, W) or (H,W)
        if mel_spec_norm.ndim == 2: # (H, W)
            pil_image = Image.fromarray((mel_spec_norm.cpu().numpy() * 255).astype(np.uint8), mode='L')
        elif mel_spec_norm.ndim == 3 and mel_spec_norm.shape[0] == 1: # (1, H, W)
            pil_image = Image.fromarray((mel_spec_norm.squeeze(0).cpu().numpy() * 255).astype(np.uint8), mode='L')
        else: # Should not happen if mono
             raise ValueError("Spectrogram has unexpected dimensions for PIL conversion.")

        # ViTImageProcessor will handle resizing and channel replication if needed (e.g. L to RGB)
        # Or, we can explicitly convert to RGB here.
        # ViT expects 3 channels typically.
        pil_image_rgb = pil_image.convert("RGB")
        return pil_image_rgb

    def get_patches_for_mae(self, video_path: str, device: torch.device):
        """
        Processes audio from a video file for MAE.
        Returns:
            all_patch_embeddings: (1, num_audio_patches, vit_hidden_size) - Embeddings from ViT's last_hidden_state.
            original_spectrogram_patches: (1, num_audio_patches, 1*patch_H*patch_W) - Flattened original spectrogram values.
            total_audio_patches: Integer, total number of patches from the spectrogram.
        """
        waveform, sr = self._extract_audio_waveform(video_path)
        if waveform is None or sr is None:
            print(f"Using zero tensors for MAE audio due to extraction failure from {video_path}")
            return (torch.zeros(1, self.mae_num_patches, self.vit_hidden_size, device=device),
                    torch.zeros(1, self.mae_num_patches, self.mae_patch_spectrogram_dim, device=device),
                    self.mae_num_patches)

        waveform = waveform.to(device) # Move waveform to the target device
        mel_spectrogram_raw = self.mel_spectrogram_transform(waveform) # (1, n_mels, Time)
        if mel_spectrogram_raw.shape[0] == 1:
            mel_spectrogram_raw = mel_spectrogram_raw.squeeze(0) # (n_mels, Time)
        
        # Preprocess to PIL Image (which handles normalization and conversion to L mode initially)
        # This image is then converted to RGB for ViTImageProcessor if not already.
        pil_image_for_processor = self._preprocess_spectrogram(mel_spectrogram_raw) # PIL Image, RGB

        # 1. Get original spectrogram patches for MAE target (single channel)
        # We need the spectrogram data *before* it's converted to 3 channels for ViT,
        # but *after* it's resized to target_spectrogram_height/width.
        # ViTImageProcessor.preprocess can give us the resized tensor.
        # We want the single-channel version for reconstruction targets.
        
        # Get single-channel normalized tensor, resized to ViT input dimensions
        # First, get the image_processor to resize it, then we take one channel.
        processed_for_vit_input = self.image_processor(images=pil_image_for_processor, return_tensors="pt")
        # pixel_values is (1, 3, H_vit, W_vit) for a single spectrogram image
        # We take one channel for the MAE target and normalize it (if not already done appropriately by _preprocess_spectrogram for single channel data)
        # The _preprocess_spectrogram normalizes the single channel mel_spec_db before converting to PIL.
        # So, we can try to get the L-mode image first, then tensor, then patch.

        pil_image_l_mode = pil_image_for_processor.convert('L')
        # Resize this L-mode image to ViT's expected input dimensions
        pil_image_l_resized = pil_image_l_mode.resize((self.target_spectrogram_width, self.target_spectrogram_height), Image.Resampling.LANCZOS)
        target_spec_tensor_l = torch.tensor(np.array(pil_image_l_resized), dtype=torch.float32).unsqueeze(0).to(device) / 255.0 # (1, H_vit, W_vit)
        
        ps = self.mae_patch_size
        H_vit, W_vit = target_spec_tensor_l.shape[1], target_spec_tensor_l.shape[2]
        # (1, num_patches_h, ps, num_patches_w, ps)
        patches_unfold = target_spec_tensor_l.unfold(1, ps, ps).unfold(2, ps, ps)
        # (1, num_patches_h, num_patches_w, ps, ps)
        patches_permute = patches_unfold.permute(0, 1, 3, 2, 4).contiguous()
        # (num_patches_h * num_patches_w, 1 * ps * ps)
        original_spectrogram_patches = patches_permute.view(1, -1, 1 * ps * ps) # Keep channel dim for consistency (1*ps*ps)

        # 2. Get patch embeddings from ViT (last_hidden_state)
        # The ViT expects a 3-channel image. inputs_for_vit is already (1, 3, H, W)
        inputs_for_vit = processed_for_vit_input['pixel_values'].to(device)

        with torch.no_grad():
            vit_outputs = self.vit_model(pixel_values=inputs_for_vit)
        
        # last_hidden_state shape: (1, num_patches_vit + 1, hidden_size)
        all_patch_embeddings = vit_outputs.last_hidden_state[:, 1:, :] # Exclude CLS

        assert all_patch_embeddings.shape[1] == self.mae_num_patches, \
            f"Audio MAE: Mismatch in num_patches for embeddings: expected {self.mae_num_patches}, got {all_patch_embeddings.shape[1]}"
        assert original_spectrogram_patches.shape[1] == self.mae_num_patches, \
            f"Audio MAE: Mismatch in num_patches for original spec: expected {self.mae_num_patches}, got {original_spectrogram_patches.shape[1]}"

        return all_patch_embeddings, original_spectrogram_patches, self.mae_num_patches

    def forward(self, video_paths_batch: list[str], device: torch.device) -> torch.Tensor:
        """
        Processes a BATCH of video files to extract audio features for contrastive learning (Stage 2).
        Extracts audio, creates spectrogram image, processes with ViT, and returns 
        patch embeddings suitable for the SharedTransformerBackbone.
        Args:
            video_paths_batch: List of paths to video files.
            device: Torch device.
        Returns:
            A tensor of patch embeddings shape (batch_size, num_audio_patches, vit_hidden_size)
            Returns zeros tensor if batch fails.
        """
        batch_size = len(video_paths_batch)
        batch_embeddings_list = []
        processed_successfully = 0
        
        for video_path in video_paths_batch:
            waveform, sr = self._extract_audio_waveform(video_path)
            if waveform is None or sr is None:
                print(f"Skipping audio processing for {video_path} due to extraction failure.")
                continue # Skip this item

            try:
                waveform = waveform.to(device) # Move waveform to the target device
                mel_spectrogram_raw = self.mel_spectrogram_transform(waveform) 
                if mel_spectrogram_raw.shape[0] == 1:
                    mel_spectrogram_raw = mel_spectrogram_raw.squeeze(0) 

                spectrogram_image = self._preprocess_spectrogram(mel_spectrogram_raw) # PIL Image, RGB
                
                inputs = self.image_processor(images=spectrogram_image, return_tensors="pt").to(device)
                inputs_for_vit = inputs['pixel_values'] # Shape (1, 3, H, W)
                
                # Pass spectrogram image through ViT
                # with torch.no_grad() if not self.training else nullcontext():
                vit_outputs = self.vit_model(pixel_values=inputs_for_vit)
                
                # Get patch embeddings (excluding CLS token from ViT internal embeddings)
                # last_hidden_state shape: (1, num_patches_vit + 1, hidden_size)
                patch_embeddings = vit_outputs.last_hidden_state[:, 1:, :] # (1, num_audio_patches, hidden_size)

                # Basic check
                if patch_embeddings.shape[1] != self.mae_num_patches:
                    print(f"Warning: Audio patch count mismatch for {video_path}. Expected {self.mae_num_patches}, got {patch_embeddings.shape[1]}. Skipping.")
                    continue # Skip this item

                batch_embeddings_list.append(patch_embeddings)
                processed_successfully += 1

            except Exception as e:
                print(f"Error processing audio for {video_path} in forward pass: {e}. Skipping.")
                continue

        if processed_successfully == 0 and batch_size > 0:
             print(f"Error: All audio processing failed for the batch in AudioModality forward.")
             # Return zeros tensor matching expected output shape for B=batch_size
             return torch.zeros(batch_size, self.mae_num_patches, self.vit_hidden_size, device=device)
        elif not batch_embeddings_list:
             return torch.zeros(0, self.mae_num_patches, self.vit_hidden_size, device=device)

        # Concatenate the list of (1, N, D) tensors into (B_processed, N, D)
        final_batch_embeddings = torch.cat(batch_embeddings_list, dim=0)
        
        return final_batch_embeddings 


if __name__ == '__main__':
    print("Running AudioModality example...")
    
    dummy_video_path = "test_video.mp4" 
    # Removed the call to create_dummy_video here.
    # The script will now assume test_video.mp4 exists or handle its absence below.

    if not os.path.exists(dummy_video_path):
        print(f"Dummy video {dummy_video_path} not found. Skipping AudioModality example run.")
        print("Please run video_modality.py first or provide a valid video file.")
    else:
        try:
            audio_processor = AudioModality(
                sample_rate=16000,
                n_mels=128,
                target_spectrogram_height=224, # ViT standard
                target_spectrogram_width=224,  # ViT standard
                output_embed_dim=512 # Example dimension
            )
            audio_processor.eval() # Set to evaluation mode
            
            print(f"Processing audio from video: {dummy_video_path}")
            audio_features = audio_processor(dummy_video_path)
            
            print(f"Output audio feature shape: {audio_features.shape}") 
            # Expected: (1, num_patches (e.g. 196 for 224x224 input and 16x16 patch), output_embed_dim (e.g. 512))
            if audio_features.nelement() > 0 : # check if not zero tensor from failure
                 print("First audio feature vector (first 5 elements of first patch):", audio_features[0, 0, :5].tolist())

            # Test with a non-existent video
            print("\nProcessing non-existent video for audio...")
            non_existent_video_features = audio_processor("non_existent_video_for_audio.mp4")
            print(f"Output for non-existent video (audio features shape): {non_existent_video_features.shape}")
            # Expected to be a zero tensor of correct shape

            # --- Test get_patches_for_mae ---
            print("\n--- Testing get_patches_for_mae for Audio ---")
            if os.path.exists(dummy_video_path):
                # audio_processor instance from above should be fine
                audio_processor.eval()
                current_device = next(audio_processor.parameters()).device
                print(f"Using device for Audio MAE test: {current_device}")
                try:
                    embeddings, original_patches, total_patches = audio_processor.get_patches_for_mae(dummy_video_path, device=current_device)
                    print(f"  Audio MAE Patch Embeddings Shape: {embeddings.shape}")
                    # Expected: (1, mae_num_patches, vit_hidden_size)
                    # e.g., (1, (224/16)^2 = 196, 768)
                    print(f"  Audio MAE Original Spectrogram Patches Shape: {original_patches.shape}")
                    # Expected: (1, mae_num_patches, 1 * patch_size * patch_size)
                    # e.g., (1, 196, 1*16*16=256)
                    print(f"  Audio MAE Total Patches: {total_patches}")
                    # Expected: 196

                    # Verify derived MAE dimensions
                    print(f"  Config: mae_patch_size={audio_processor.mae_patch_size}")
                    print(f"  Config: mae_num_patches={audio_processor.mae_num_patches}")
                    print(f"  Config: mae_patch_spectrogram_dim={audio_processor.mae_patch_spectrogram_dim}")

                except Exception as e_mae_audio:
                    print(f"Error during Audio get_patches_for_mae test: {e_mae_audio}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Skipping Audio get_patches_for_mae test as dummy video not found.")
            # --- End Test get_patches_for_mae ---

            # --- Test Batch Forward ---
            print("\n--- Testing Audio Batch Forward (Stage 2 features) ---")
            test_batch_paths_audio = []
            # Reuse paths from video test if they exist
            if os.path.exists("test_video.mp4"): test_batch_paths_audio.append("test_video.mp4")
            if os.path.exists("test_video_main_model_0.mp4"): test_batch_paths_audio.append("test_video_main_model_0.mp4")
            if os.path.exists("test_video_main_model_1.mp4"): test_batch_paths_audio.append("test_video_main_model_1.mp4")
            
            if len(test_batch_paths_audio) >= 1:
                print(f"Processing batch of {len(test_batch_paths_audio)} videos for audio...")
                try:
                    batch_features_audio = audio_processor(test_batch_paths_audio, device=current_device)
                    print(f"  Audio Batch Forward Output Shape: {batch_features_audio.shape}")
                    # Expected: (len(test_batch_paths_audio), mae_num_patches, vit_hidden_size)
                    # e.g., (B, 196, 768)
                except Exception as e_fwd_audio:
                    print(f"Error during audio batch forward test: {e_fwd_audio}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Skipping audio batch forward test as no suitable test videos found.")
            # --- End Test Batch Forward ---

        except ImportError as e:
            print(f"Import error during AudioModality example: {e}.")
            print("Please ensure all dependencies (torch, torchaudio, transformers, moviepy, numpy, Pillow) are installed.")
        except Exception as e:
            print(f"An error occurred during the AudioModality example run: {e}")
            import traceback
            traceback.print_exc()

    print("\nAudioModality example finished.") 