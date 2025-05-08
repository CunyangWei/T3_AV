import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
from tqdm import tqdm
import numpy as np
from transformers import get_cosine_schedule_with_warmup

# Import project modules
from t3_av_model import T3_AV_Model
from vggsound_dataset import VGGSoundDataset, vggsound_collate_fn

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Contrastive Learning for T3-AV Model")

    # --- Data Paths --- 
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the vggsound.csv file.')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to the directory containing video files.')
    parser.add_argument('--stage1_checkpoint', type=str, required=True, help='Path to the stage 1 pretrained checkpoint (.pt file).')
    parser.add_argument('--output_dir', type=str, default='./stage2_output', help='Directory to save stage 2 checkpoints and logs.')

    # --- Training Hyperparameters --- 
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs for stage 2.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for stage 2 training.')
    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-5, help='Peak learning rate for stage 2.')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for AdamW optimizer.')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs for learning rate scheduler.')
    
    # --- Contrastive Learning Specific --- 
    parser.add_argument('--contrastive_dim', type=int, default=128, help='Embedding dimension for contrastive loss.')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature scaling for InfoNCE loss.')
    
    # --- Model Freezing Options --- 
    parser.add_argument('--freeze_encoders', action='store_true', help='Freeze the video and audio modality encoders.')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze the shared transformer backbone.')

    # --- Model Configuration (should ideally match stage 1, but allow overrides) --- 
    # These might be loaded from the checkpoint args, but providing them allows flexibility
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames sampled per video (must match stage 1).')
    parser.add_argument('--vit_model_name', type=str, default='google/vit-base-patch16-224-in21k', help='Pretrained ViT model name (must match stage 1).')
    parser.add_argument('--audio_duration', type=float, default=10.0, help='Duration of audio clip in seconds (must match stage 1).')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate (must match stage 1).')
    parser.add_argument('--n_mels', type=int, default=128, help='Number of Mel bands (must match stage 1).')
    parser.add_argument('--bb_layers', type=int, default=6, help='Number of layers in the shared backbone.')
    parser.add_argument('--bb_heads', type=int, default=12, help='Number of attention heads in the shared backbone.')
    # Note: MAE decoder params not needed here, but proj head params are.
    parser.add_argument('--proj_hidden_dim', type=int, default=768, help='Hidden dimension for projection head MLP.') # Example
    # Add dummy clf_hidden_dim needed for T3_AV_Model init
    parser.add_argument('--clf_hidden_dim', type=int, default=512, 
                        help='Dummy hidden dimension for classification head (Stage 3). Not used in Stage 2.')
    # Dummy params for MAE decoders if T3_AV_Model requires them in __init__
    parser.add_argument('--dec_layers', type=int, default=2, help='Dummy MAE decoder layers.') # Added for consistency
    parser.add_argument('--dec_heads', type=int, default=8, help='Dummy MAE decoder heads.') 
    # dec_mlp_dim often derived

    # --- System Configuration --- 
    parser.add_argument('--device', type=str, default='cuda', help="Device to use ('cuda' or 'cpu').")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for DataLoader.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--log_interval', type=int, default=50, help='Log training status every N steps.')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Save checkpoint every N epochs.')

    args = parser.parse_args()
    return args

def main(args):
    # --- Setup --- 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, using CPU.")
        args.device = 'cpu'
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Using device: {device}")

    # --- Model Configuration --- 
    # Reconstruct necessary model config based on args (should match stage 1 setup)
    video_modality_params = {'num_frames_to_sample': args.num_frames, 'pretrained_vit_name': args.vit_model_name, 'output_embed_dim': 768}
    audio_modality_params = {
        'sample_rate': args.sample_rate, 'n_mels': args.n_mels,
        'target_spectrogram_height': 224, 'target_spectrogram_width': 224,
        'pretrained_vit_name': args.vit_model_name, 'audio_duration_secs': args.audio_duration,
        'n_fft': 1024, 'hop_length': 512, 'output_embed_dim': 768
    }
    shared_backbone_params = {'num_heads': args.bb_heads, 'num_layers': args.bb_layers, 'mlp_dim': 768 * 4, 'dropout': 0.1}
    # MAE decoders not strictly needed but T3_AV_Model expects the params
    mae_decoder_base_params = {'num_heads': args.dec_heads, 'num_decoder_layers': args.dec_layers, 'mlp_dim': 768 * 4, 'dropout': 0.1}
    proj_head_base_params = {'hidden_dim': args.proj_hidden_dim, 'use_gelu': True}
    # Dummy sound_clf_head_params required by T3_AV_Model init
    sound_clf_head_params = {'hidden_dim': args.clf_hidden_dim, 'num_classes': 1, 'use_gelu': True}

    # --- Initialize Model --- 
    print("Initializing T3_AV_Model for Stage 2...")
    model = T3_AV_Model(
        video_modality_params=video_modality_params,
        audio_modality_params=audio_modality_params,
        shared_backbone_params=shared_backbone_params,
        video_mae_decoder_base_params=mae_decoder_base_params, # Pass dummy/base params
        audio_mae_decoder_base_params=mae_decoder_base_params, # Pass dummy/base params
        proj_head_base_params=proj_head_base_params,
        sound_clf_head_params=sound_clf_head_params,          # Added
        contrastive_embed_dim=args.contrastive_dim,
        contrastive_temperature=args.temperature,
        # mae_mask_ratio not used in stage 2 forward, but __init__ needs it
        mae_mask_ratio=0.75 
    )
    print("Model structure initialized.")

    # --- Load Stage 1 Checkpoint --- 
    if not os.path.exists(args.stage1_checkpoint):
        print(f"Error: Stage 1 checkpoint not found at {args.stage1_checkpoint}")
        return
    
    print(f"Loading Stage 1 checkpoint from: {args.stage1_checkpoint}")
    checkpoint = torch.load(args.stage1_checkpoint, map_location='cpu')
    
    # Get the state dict from the checkpoint
    checkpoint_state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Remove the classifier weights (if they exist in the checkpoint) 
    # as they have a different number of classes (dummy 1 vs actual N).
    keys_to_remove = [k for k in checkpoint_state_dict if k.startswith('sound_classifier.')]
    # Also remove the positional encoding buffer due to potential size mismatch from different max_seq_len calculations
    # between saving (Stage 1 - unmasked length) and loading (Stage 2 - full length).
    pe_key = 'shared_backbone.pos_encoder.pe'
    if pe_key in checkpoint_state_dict:
        keys_to_remove.append(pe_key)

    if keys_to_remove:
        print(f"Removing keys from checkpoint before loading: {keys_to_remove}")
        for key in keys_to_remove:
            if key in checkpoint_state_dict: # Check again in case key wasn't present
                del checkpoint_state_dict[key]
            else:
                print(f"  Warning: Attempted to remove key '{key}' but it was not found in checkpoint state_dict.")
    
    try:
        load_result = model.load_state_dict(checkpoint_state_dict, strict=False)
        print(f"Checkpoint load result: {load_result}")
        # Check for missing/unexpected keys if needed
        if load_result.missing_keys:
            print("Missing keys:", load_result.missing_keys) # Expected: proj head keys
        if load_result.unexpected_keys:
            print("Warning: Unexpected keys found:", load_result.unexpected_keys)
    except KeyError:
        print("Error: 'model_state_dict' not found in checkpoint. Attempting to load the whole checkpoint as state_dict.")
        try:
           load_result = model.load_state_dict(checkpoint, strict=False)
           print(f"Checkpoint load result (alternate attempt): {load_result}")
        except Exception as e:
           print(f"Error: Could not load state dict from checkpoint: {e}")
           return
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return
        
    model = model.to(device)
    print("Stage 1 weights loaded into model.")

    # --- Freezing Components --- 
    if args.freeze_encoders:
        print("Freezing Modality Encoders (Video & Audio)...")
        for name, param in model.video_encoder.named_parameters():
            param.requires_grad = False
        for name, param in model.audio_encoder.named_parameters():
            param.requires_grad = False
            
    if args.freeze_backbone:
        print("Freezing Shared Transformer Backbone...")
        for name, param in model.shared_backbone.named_parameters():
            param.requires_grad = False
            
    # Verify which parameters are trainable
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_trainable_params:,} trainable / {num_total_params:,} total")
    if num_trainable_params == 0:
        print("Warning: No parameters are set to trainable. Check freezing flags.")
        # return # Exit if nothing to train

    # --- Initialize Dataset & DataLoader --- 
    print("Initializing Dataset and DataLoader for Stage 2...")
    train_dataset = VGGSoundDataset(csv_file_path=args.csv_path, video_dir=args.video_dir, split='train')
    if len(train_dataset) == 0:
        print("Error: Training dataset is empty.")
        return
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=vggsound_collate_fn, 
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    print("Dataset and DataLoader initialized.")

    # --- Initialize Optimizer & Scheduler --- 
    # Filter parameters to only optimize those that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print("Error: No trainable parameters found for the optimizer.")
        return
        
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    print("Optimizer and Scheduler initialized for trainable parameters.")
    
    # --- Save Stage 2 Configuration --- 
    config_path = os.path.join(args.output_dir, 'stage2_config.json')
    with open(config_path, 'w') as f:
        # Include loaded checkpoint path in saved config
        args_dict = vars(args)
        args_dict['loaded_stage1_checkpoint'] = args.stage1_checkpoint
        json.dump(args_dict, f, indent=4)
    print(f"Saved Stage 2 configuration to {config_path}")

    # --- Training Loop --- 
    print("Starting Stage 2 Contrastive Training...")
    for epoch in range(args.epochs):
        model.train() # Set model to training mode
        epoch_contrastive_loss = 0.0
        processed_samples_in_epoch = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        
        for step, batch in enumerate(progress_bar):
            video_paths = batch['video_paths_batch']
            # labels = batch['labels_batch'].to(device) # Labels not used in stage 2 loss
            
            if not video_paths: continue

            optimizer.zero_grad()
            
            try:
                 # Forward pass for Stage 2
                 contrastive_loss, batch_size_processed = model.forward_stage2(
                     video_paths_batch=video_paths, 
                     device=device
                 )
            except Exception as e:
                 print(f"\nError during forward pass stage 2 at step {step} in epoch {epoch+1}: {e}")
                 print(f"Video paths in failing batch: {video_paths}")
                 print("Skipping this batch.")
                 continue 
                 
            if contrastive_loss is None or not torch.isfinite(contrastive_loss) or batch_size_processed == 0:
                 print(f"\nWarning: Non-finite loss ({contrastive_loss}) or zero processed samples ({batch_size_processed}) at step {step}. Skipping batch.")
                 continue 
                 
            # Backward pass & optimization
            contrastive_loss.backward()
            optimizer.step()
            scheduler.step()
            
            # --- Logging --- 
            current_lr = optimizer.param_groups[0]['lr']
            loss_item = contrastive_loss.item()
            epoch_contrastive_loss += loss_item * batch_size_processed # Accumulate loss weighted by processed samples
            processed_samples_in_epoch += batch_size_processed
            
            log_msg = f'Contrastive Loss: {loss_item:.4f}, LR: {current_lr:.6f}'
            progress_bar.set_postfix_str(log_msg)
                
        # --- End of Epoch --- 
        avg_epoch_loss = epoch_contrastive_loss / processed_samples_in_epoch if processed_samples_in_epoch > 0 else 0
        print(f"\nEpoch {epoch+1} Finished. Average Contrastive Loss: {avg_epoch_loss:.4f}")

        # --- Checkpointing --- 
        if (epoch + 1) % args.checkpoint_interval == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f't3_av_stage2_epoch_{epoch+1}.pt')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': vars(args) # Store args used for this checkpoint
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Stage 2 Contrastive Training finished.")

if __name__ == '__main__':
    import numpy as np 
    args = parse_args()
    main(args) 