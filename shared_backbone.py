import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x is batch_first, pe is (max_len, 1, d_model) -> (seq_len, 1, d_model)
        x = x + self.pe[:x.size(1)].transpose(0,1) # self.pe is [seq_len, 1, d_model], need [1, seq_len, d_model] for broadcast
        return self.dropout(x)

class SharedTransformerBackbone(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, mlp_dim: int, dropout: float = 0.1, max_seq_len: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Positional encoding for the sequence including the CLS token
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=max_seq_len + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=mlp_dim, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src_tokens: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src_tokens: (batch_size, seq_len, embed_dim)
            src_key_padding_mask: (batch_size, seq_len) - NOT for CLS token, for original seq
        Returns:
            (batch_size, seq_len + 1, embed_dim)
        """
        batch_size = src_tokens.size(0)
        cls_tokens_expanded = self.cls_token.expand(batch_size, -1, -1) # (batch_size, 1, embed_dim)
        
        # Prepend CLS token
        tokens_with_cls = torch.cat((cls_tokens_expanded, src_tokens), dim=1) # (batch_size, seq_len + 1, embed_dim)
        
        # Add positional encoding
        tokens_with_cls_pos = self.pos_encoder(tokens_with_cls)

        # Adjust padding mask for CLS token if provided
        if src_key_padding_mask is not None:
            # Add a False (not masked) for the CLS token at the beginning of each sequence in the mask
            cls_padding = torch.full((batch_size, 1), False, dtype=torch.bool, device=src_key_padding_mask.device)
            full_padding_mask = torch.cat((cls_padding, src_key_padding_mask), dim=1)
        else:
            full_padding_mask = None
            
        output = self.transformer_encoder(tokens_with_cls_pos, src_key_padding_mask=full_padding_mask)
        return output

class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, use_gelu: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU() if use_gelu else nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, backbone_output_sequence: torch.Tensor, use_cls_token: bool = True) -> torch.Tensor:
        """
        Args:
            backbone_output_sequence: (batch_size, seq_len_plus_cls, embed_dim)
            use_cls_token: If True, uses the first token (CLS). Otherwise, averages other tokens.
        """
        if use_cls_token:
            feature = backbone_output_sequence[:, 0]  # CLS token representation
        else:
            # Average pooling of patch tokens (excluding CLS token)
            feature = backbone_output_sequence[:, 1:].mean(dim=1)
        
        x = self.fc1(feature)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_decoder_layers: int, mlp_dim: int, 
                 output_patch_dim: int, max_total_patches: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_patch_dim = output_patch_dim
        self.max_total_patches = max_total_patches

        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_encoder_decoder = PositionalEncoding(embed_dim, dropout, max_len=max_total_patches)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=mlp_dim, 
            dropout=dropout, 
            batch_first=True
        ) # Using EncoderLayer for simplicity as common in MAE decoders
        self.transformer_decoder_core = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.output_projection = nn.Linear(embed_dim, output_patch_dim)

    def forward(self, unmasked_features_from_backbone: torch.Tensor, 
                  unmasked_indices: torch.LongTensor, 
                  masked_indices: torch.LongTensor, 
                  total_patches_in_sequence: int) -> torch.Tensor:
        """
        Args:
            unmasked_features_from_backbone: (batch_size, num_unmasked_tokens, embed_dim)
                                             These are outputs from shared backbone FOR UNMASKED PATCHES.
            unmasked_indices: (batch_size, num_unmasked_tokens) - original positions of unmasked patches.
            masked_indices: (batch_size, num_masked_tokens) - original positions of masked patches.
            total_patches_in_sequence: The total number of patches in the original full sequence (N_v or N_a).
        Returns:
            reconstructed_patches: (batch_size, total_patches_in_sequence, output_patch_dim)
        """
        batch_size = unmasked_features_from_backbone.size(0)
        num_unmasked = unmasked_indices.size(1)
        num_masked = masked_indices.size(1)

        assert total_patches_in_sequence <= self.max_total_patches, \
            f"total_patches_in_sequence ({total_patches_in_sequence}) > max_total_patches ({self.max_total_patches})"
        assert num_unmasked + num_masked == total_patches_in_sequence, \
            "Sum of unmasked and masked tokens must equal total_patches_in_sequence"

        # Initialize full sequence with placeholder for scatter (doesn't matter what, will be overwritten or is mask)
        full_sequence_embed = self.mask_token.expand(batch_size, total_patches_in_sequence, self.embed_dim).clone()

        # Scatter unmasked features to their original positions
        # unmasked_indices needs to be (batch_size, num_unmasked, 1) and expanded for gather/scatter
        idx_expand = unmasked_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        full_sequence_embed.scatter_(dim=1, index=idx_expand, src=unmasked_features_from_backbone)
        
        # Add positional encoding to the full sequence (unmasked features + mask tokens)
        full_sequence_pos = self.pos_encoder_decoder(full_sequence_embed)
        
        # Pass through Transformer decoder core
        decoded_features = self.transformer_decoder_core(full_sequence_pos)
        
        # Project to patch dimension
        reconstructed_patches = self.output_projection(decoded_features) # (B, total_patches, output_patch_dim)
        return reconstructed_patches

class SoundClassificationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, use_gelu: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU() if use_gelu else nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes) # Output logits

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor, shape [batch_size, input_dim] or [batch_size, seq_len_plus_cls, input_dim]
                      If 3D, the CLS token (features[:, 0]) is used.
                      If 2D, the features tensor is used directly.
        """
        if features.ndim == 3:
            feature_to_use = features[:, 0]  # CLS token representation from a sequence
        elif features.ndim == 2:
            feature_to_use = features # Already a prepared feature vector (e.g., fused CLS tokens)
        else:
            raise ValueError(f"SoundClassificationHead expects a 2D or 3D tensor, got {features.ndim}D")
        
        x = self.fc1(feature_to_use)
        x = self.activation(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    print("Testing Shared Backbone Components...")
    # --- Configuration ---
    batch_s = 4
    video_seq_len = 16 # num_frames_to_sample from VideoModality
    audio_seq_len = 196 # num_patches from AudioModality (e.g., (224/16)**2)
    modal_embed_dim = 768 # Should match output_embed_dim of Video/AudioModality
    max_overall_seq_len = max(video_seq_len, audio_seq_len) # For shared backbone pos encoding

    # Shared Backbone_config
    bb_heads = 12
    bb_layers = 6
    bb_mlp_dim = modal_embed_dim * 4 

    # Projection Head config
    proj_hidden_dim = modal_embed_dim
    proj_output_dim = 128

    # MAE Decoder config
    mae_dec_layers = 2
    mae_dec_heads = 8
    video_patch_dim = 16 * 16 * 3 # Example for 16x16 RGB patches
    audio_patch_dim = 16 * 16 * 1 # Example for 16x16 spectrogram patches (mono)
    max_mae_patches = audio_seq_len # Max patches for MAE decoder pos encoding (e.g. audio has more)

    # Classification Head config
    clf_hidden_dim = 512
    num_sound_classes = 309 # Example from VGGSound (can vary)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate Modules ---
    print("\n--- Instantiating Modules ---")
    shared_backbone = SharedTransformerBackbone(
        embed_dim=modal_embed_dim, num_heads=bb_heads, num_layers=bb_layers, 
        mlp_dim=bb_mlp_dim, max_seq_len=max_overall_seq_len
    ).to(device)
    print("SharedTransformerBackbone instantiated.")

    video_projection_head = ProjectionHead(modal_embed_dim, proj_hidden_dim, proj_output_dim).to(device)
    audio_projection_head = ProjectionHead(modal_embed_dim, proj_hidden_dim, proj_output_dim).to(device)
    print("ProjectionHeads instantiated.")

    video_mae_decoder = MAEDecoder(
        embed_dim=modal_embed_dim, num_heads=mae_dec_heads, num_decoder_layers=mae_dec_layers,
        mlp_dim=bb_mlp_dim, output_patch_dim=video_patch_dim, max_total_patches=video_seq_len
    ).to(device)
    audio_mae_decoder = MAEDecoder(
        embed_dim=modal_embed_dim, num_heads=mae_dec_heads, num_decoder_layers=mae_dec_layers,
        mlp_dim=bb_mlp_dim, output_patch_dim=audio_patch_dim, max_total_patches=audio_seq_len
    ).to(device)
    print("MAEDecoders instantiated.")

    sound_classifier = SoundClassificationHead(
        modal_embed_dim, clf_hidden_dim, num_sound_classes
    ).to(device)
    print("SoundClassificationHead instantiated.")

    # --- Dummy Inputs ---
    print("\n--- Testing with Dummy Inputs ---")
    dummy_video_features = torch.randn(batch_s, video_seq_len, modal_embed_dim).to(device)
    dummy_audio_features = torch.randn(batch_s, audio_seq_len, modal_embed_dim).to(device)
    # Dummy padding mask (e.g., last 2 tokens of video are padding)
    dummy_video_padding_mask = torch.zeros(batch_s, video_seq_len, dtype=torch.bool, device=device)
    if video_seq_len > 2: dummy_video_padding_mask[:, -2:] = True

    # 1. Test Shared Backbone
    print("\n1. Shared Backbone Processing:")
    video_bb_out = shared_backbone(dummy_video_features, src_key_padding_mask=dummy_video_padding_mask)
    audio_bb_out = shared_backbone(dummy_audio_features) # No padding mask for audio example
    print(f"  Video Backbone Output Shape: {video_bb_out.shape}") # Expected: (B, video_seq_len+1, modal_embed_dim)
    print(f"  Audio Backbone Output Shape: {audio_bb_out.shape}") # Expected: (B, audio_seq_len+1, modal_embed_dim)

    # 2. Test Projection Heads
    print("\n2. Projection Head Processing (using CLS token):")
    video_projected = video_projection_head(video_bb_out)
    audio_projected = audio_projection_head(audio_bb_out)
    print(f"  Video Projected Shape: {video_projected.shape}") # Expected: (B, proj_output_dim)
    print(f"  Audio Projected Shape: {audio_projected.shape}") # Expected: (B, proj_output_dim)

    # 3. Test MAE Decoders
    print("\n3. MAE Decoder Processing:")
    # For Video MAE:
    num_unmasked_video = video_seq_len // 2
    num_masked_video = video_seq_len - num_unmasked_video
    # These are features of UNMASKED patches from the backbone (excluding CLS token's output)
    unmasked_video_bb_patch_features = video_bb_out[:, 1:num_unmasked_video+1, :] 
    dummy_unmasked_video_indices = torch.sort(torch.randperm(video_seq_len)[:num_unmasked_video].expand(batch_s, -1), dim=1)[0].to(device)
    dummy_masked_video_indices = torch.sort(torch.randperm(video_seq_len)[num_unmasked_video:].expand(batch_s, -1), dim=1)[0].to(device)
    # Ensure indices cover the whole range and are distinct for this dummy example
    # This dummy index generation is simplified; a real MAE would have proper shuffling and splitting.
    if num_unmasked_video + num_masked_video == video_seq_len:
        reconstructed_video_patches = video_mae_decoder(
            unmasked_video_bb_patch_features, 
            dummy_unmasked_video_indices, 
            dummy_masked_video_indices, 
            video_seq_len
        )
        print(f"  Reconstructed Video Patches Shape: {reconstructed_video_patches.shape}") # (B, video_seq_len, video_patch_dim)
    else:
        print("  Skipping Video MAE due to dummy index mismatch.")

    # For Audio MAE:
    num_unmasked_audio = audio_seq_len // 2
    num_masked_audio = audio_seq_len - num_unmasked_audio
    unmasked_audio_bb_patch_features = audio_bb_out[:, 1:num_unmasked_audio+1, :]
    dummy_unmasked_audio_indices = torch.sort(torch.randperm(audio_seq_len)[:num_unmasked_audio].expand(batch_s, -1), dim=1)[0].to(device)
    dummy_masked_audio_indices = torch.sort(torch.randperm(audio_seq_len)[num_unmasked_audio:].expand(batch_s, -1), dim=1)[0].to(device)
    if num_unmasked_audio + num_masked_audio == audio_seq_len:
        reconstructed_audio_patches = audio_mae_decoder(
            unmasked_audio_bb_patch_features, 
            dummy_unmasked_audio_indices, 
            dummy_masked_audio_indices, 
            audio_seq_len
        )
        print(f"  Reconstructed Audio Patches Shape: {reconstructed_audio_patches.shape}") # (B, audio_seq_len, audio_patch_dim)
    else:
        print("  Skipping Audio MAE due to dummy index mismatch.")

    # 4. Test Sound Classification Head
    print("\n4. Sound Classification Head Processing (using CLS from audio path):")
    sound_logits = sound_classifier(audio_bb_out)
    print(f"  Sound Logits Shape: {sound_logits.shape}") # Expected: (B, num_sound_classes)

    print("\nComponent tests finished.") 