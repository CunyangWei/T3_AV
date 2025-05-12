import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import json
from transformers import get_cosine_schedule_with_warmup

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Import project modules
from t3_av_model import T3_AV_Model
from vggsound_dataset import VGGSoundDataset, vggsound_collate_fn

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 MAE Pre-training for T3-AV Model")

    # --- Data Paths --- 
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the vggsound.csv file.')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to the directory containing video files.')
    parser.add_argument('--output_dir', type=str, default='./stage1_output', help='Directory to save checkpoints and logs.')

    # --- Training Hyperparameters --- 
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', '--lr', type=float, default=1.5e-4, help='Peak learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for AdamW optimizer.')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs for learning rate scheduler.')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Proportion of patches to mask for MAE.')
    # Flags default to True, use --no_train_... to disable
    parser.add_argument('--no_train_video_mae', action='store_false', dest='train_video_mae', help='Disable video MAE training.')
    parser.add_argument('--no_train_audio_mae', action='store_false', dest='train_audio_mae', help='Disable audio MAE training.')
    parser.set_defaults(train_video_mae=True, train_audio_mae=True) # Explicitly set default
    
    # --- Model Configuration --- 
    # Video Modality Params (Simplified - Add more if needed)
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames to sample per video.')
    parser.add_argument('--vit_model_name', type=str, default='google/vit-base-patch16-224-in21k', help='Pretrained Vision Transformer model name.')
    
    # Audio Modality Params (Simplified - Add more if needed)
    parser.add_argument('--audio_duration', type=float, default=10.0, help='Duration of audio clip in seconds.')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate.')
    parser.add_argument('--n_mels', type=int, default=128, help='Number of Mel bands.')
    # n_fft, hop_length could also be args if needed

    # Shared Backbone Params (Simplified - Add more if needed)
    parser.add_argument('--bb_layers', type=int, default=6, help='Number of layers in the shared backbone transformer.')
    parser.add_argument('--bb_heads', type=int, default=12, help='Number of attention heads in the shared backbone.')
    # mlp_dim often derived from embed_dim, e.g., embed_dim * 4

    # MAE Decoder Params (Simplified - Add more if needed)
    parser.add_argument('--dec_layers', type=int, default=2, help='Number of layers in MAE decoders.')
    parser.add_argument('--dec_heads', type=int, default=8, help='Number of attention heads in MAE decoders.')
    # dec_mlp_dim often derived

    # Dummy/Placeholder parameters for heads not used in Stage 1, but required by T3_AV_Model.__init__
    parser.add_argument('--proj_hidden_dim_contrastive', type=int, default=768, 
                        help='Dummy hidden dimension for contrastive projection head (Stage 2). Not used in Stage 1.')
    parser.add_argument('--contrastive_embed_dim', type=int, default=128, 
                        help='Dummy embedding dimension for contrastive loss (Stage 2). Not used in Stage 1.')
    parser.add_argument('--clf_hidden_dim', type=int, default=512, 
                        help='Dummy hidden dimension for classification head (Stage 3). Not used in Stage 1.')
    # num_classes for clf_head will be set to a dummy value like 1 in main() as it's not relevant for stage 1

    # --- System Configuration --- 
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu).')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for DataLoader.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--log_interval', type=int, default=50, help='Log training status every N steps.')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Save checkpoint every N epochs.')

    # --- DDP Configuration ---
    parser.add_argument('--ddp', action='store_true', help='Enable Distributed Data Parallel training.')
    # local_rank is typically provided by the launch utility (e.g., torchrun) via environment variables

    args = parser.parse_args()
    
    # Basic validation
    if not args.train_video_mae and not args.train_audio_mae:
        raise ValueError("At least one of --train_video_mae or --train_audio_mae must be set.")
        
    return args

def main(args):
    # --- DDP Setup (if enabled) ---
    is_ddp = args.ddp
    rank = 0
    world_size = 1
    local_rank = 0 # Default for non-DDP or master process

    if is_ddp:
        if "LOCAL_RANK" not in os.environ:
            # Fallback if not using torchrun but ddp is flagged (less common for multi-gpu)
            # This setup is basic and might need adjustment for non-torchrun launchers.
            if torch.cuda.device_count() > 1:
                print("Warning: LOCAL_RANK not set, attempting basic DDP init. Use torchrun for robust DDP.")
                # This is a simplified setup; proper distributed launch is preferred.
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355') # Ensure port is free or unique
                # Determine rank and world_size if possible, or assume single node multi-GPU
                # This part is tricky without a proper launcher. For torchrun, LOCAL_RANK is set.
                # For simplicity, we'll rely on torchrun setting LOCAL_RANK.
                # If you must run without torchrun and with --ddp, this part needs more robust handling.
                raise RuntimeError("DDP enabled but LOCAL_RANK not set. Please use torchrun or a similar launch utility.")

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        if rank == 0: print(f"DDP enabled: {world_size} GPU(s). Rank {rank} on device cuda:{local_rank}")
    else:
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, using CPU.")
            args.device = 'cpu'
        device = torch.device(args.device)
        if rank == 0: print(f"DDP not enabled. Using device: {device}")

    # --- Setup --- 
    torch.manual_seed(args.seed + rank) 
    np.random.seed(args.seed + rank) 
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
    # Ensure all processes have `device` set before proceeding
    if is_ddp: dist.barrier() 
    if rank == 0 or not is_ddp : print(f"Process Rank {rank}: Using device: {device}")

    # --- Model Configuration --- 
    # Prepare config dicts for T3_AV_Model constructor
    video_modality_params = {
        'num_frames_to_sample': args.num_frames,
        'pretrained_vit_name': args.vit_model_name,
        # output_embed_dim is not strictly needed for MAE stage but VideoModality requires it
        # We can set it to a default or derive later if needed for other stages
        'output_embed_dim': 768 # Example default, check VideoModality if needed
    }
    audio_modality_params = {
        'sample_rate': args.sample_rate,
        'n_mels': args.n_mels,
        'target_spectrogram_height': 224, # Assuming ViT standard, could be args
        'target_spectrogram_width': 224, # Assuming ViT standard, could be args
        'pretrained_vit_name': args.vit_model_name,
        'audio_duration_secs': args.audio_duration,
        # Set reasonable defaults for other audio params if not args
        'n_fft': 1024, 
        'hop_length': 512,
        'output_embed_dim': 768 # Example default, check AudioModality if needed
    }
    # Base params for shared backbone (embed_dim derived inside T3_AV_Model)
    shared_backbone_params = {
        'num_heads': args.bb_heads,
        'num_layers': args.bb_layers,
        'mlp_dim': 768 * 4, # Assuming base ViT hidden size * 4, adjust if needed or make arg
        'dropout': 0.1 # Default dropout, could be arg
    }
    # Base params for MAE decoders (dims derived inside T3_AV_Model)
    mae_decoder_base_params = {
        'num_heads': args.dec_heads,
        'num_decoder_layers': args.dec_layers,
        'mlp_dim': 768 * 4, # Can be smaller, make arg if needed
        'dropout': 0.1
    }

    # Dummy base params for ProjectionHead (for Stage 2), required by __init__
    proj_head_base_params = {
        'hidden_dim': args.proj_hidden_dim_contrastive,
        'use_gelu': True # Default, can be made an arg if needed for other stages
    }

    # Dummy base params for SoundClassificationHead (for Stage 3), required by __init__
    # num_classes is not used in Stage 1, so set to a placeholder value.
    sound_clf_head_params = {
        'hidden_dim': args.clf_hidden_dim,
        'num_classes': 1, # Dummy value, not used in Stage 1
        'use_gelu': True # Default
    }

    # --- Initialize Model --- 
    if rank == 0: print("Initializing T3_AV_Model...")
    model = T3_AV_Model(
        video_modality_params=video_modality_params,
        audio_modality_params=audio_modality_params,
        shared_backbone_params=shared_backbone_params,
        video_mae_decoder_base_params=mae_decoder_base_params,
        audio_mae_decoder_base_params=mae_decoder_base_params,
        proj_head_base_params=proj_head_base_params,
        sound_clf_head_params=sound_clf_head_params,
        contrastive_embed_dim=args.contrastive_embed_dim,
        mae_mask_ratio=args.mask_ratio
    ).to(device) # Move model to device before DDP wrapping
    
    if rank == 0:
        print("Model initialized.")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {num_params:,} trainable parameters (before DDP wrapping).")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if rank == 0: print("Model wrapped with DDP.")

    # --- Initialize Dataset & DataLoader --- 
    if rank == 0: print("Initializing Dataset and DataLoader...")
    train_dataset = VGGSoundDataset(csv_file_path=args.csv_path, video_dir=args.video_dir, split='train')
    
    if len(train_dataset) == 0:
        if rank == 0: print("Error: Training dataset is empty. Please check CSV path, video directory, and split name.")
        if is_ddp: dist.barrier() 
        return
    
    train_sampler = None
    shuffle_dataloader = True
    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        shuffle_dataloader = False # Sampler handles shuffling
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=shuffle_dataloader, 
        collate_fn=vggsound_collate_fn, 
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'), 
        sampler=train_sampler,
        drop_last=True if is_ddp else False 
    )
    if rank == 0: print("Dataset and DataLoader initialized.")

    # --- Initialize Optimizer & Scheduler --- 
    optimizer_params = model.module.parameters() if is_ddp else model.parameters()
    optimizer = optim.AdamW(optimizer_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    num_training_steps = len(train_loader) * args.epochs 
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    
    if rank == 0: print("Optimizer and Scheduler initialized.")
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # --- Save Configuration --- 
    if rank == 0:
        config_path = os.path.join(args.output_dir, 'stage1_config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
        print(f"Saved configuration to {config_path}")

    if is_ddp: dist.barrier() # Ensure config is saved before training starts

    # --- Training Loop --- 
    if rank == 0: print(f"Starting Stage 1 MAE Training on {world_size} GPU(s)...")
    for epoch in range(args.epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch) 

        model.train() 
        epoch_loss = 0.0
        epoch_video_loss = 0.0
        epoch_audio_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Rank {rank})", leave=False, disable=(rank != 0))
        
        for step, batch in enumerate(progress_bar):
            video_paths = batch['video_paths_batch']
            # labels = batch['labels_batch'].to(device) # Labels not used in stage 1 loss
            
            if not video_paths: # Skip empty batches if they somehow occur
                continue

            optimizer.zero_grad()
            
            try:
                 # Forward pass
                 # When using DDP, access original model methods via model.module
                 if is_ddp:
                     total_loss, video_loss_val, audio_loss_val = model.module.forward_stage1(
                         video_paths_batch=video_paths, 
                         device=device, 
                         train_video_mae=args.train_video_mae,
                         train_audio_mae=args.train_audio_mae
                     )
                 else:
                     total_loss, video_loss_val, audio_loss_val = model.forward_stage1(
                         video_paths_batch=video_paths, 
                         device=device,
                         train_video_mae=args.train_video_mae,
                         train_audio_mae=args.train_audio_mae
                     )
            except Exception as e:
                 print(f"\nError during forward pass at step {step} in epoch {epoch+1} on Rank {rank}: {e}")
                 print(f"Video paths in failing batch: {video_paths}")
                 # Decide how to handle: skip batch, raise error, etc.
                 # For robustness, let's skip the batch and continue training.
                 print("Skipping this batch.")
                 continue # Skip to the next batch
                 
            if total_loss is None or not torch.isfinite(total_loss):
                 print(f"\nWarning: Non-finite loss detected at step {step} in epoch {epoch+1} on Rank {rank}. Skipping batch.")
                 print(f"  Total Loss: {total_loss}")
                 print(f"  Video MAE Loss: {video_loss_val}")
                 print(f"  Audio MAE Loss: {audio_loss_val}")
                 continue # Skip batch if loss is NaN or Inf
                 
            # Backward pass & optimization
            total_loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # --- Logging --- 
            current_lr = optimizer.param_groups[0]['lr']
            epoch_loss += total_loss.item()
            if video_loss_val is not None: epoch_video_loss += (video_loss_val.item() if torch.is_tensor(video_loss_val) else video_loss_val)
            if audio_loss_val is not None: epoch_audio_loss += (audio_loss_val.item() if torch.is_tensor(audio_loss_val) else audio_loss_val)
            num_batches += 1
            
            if rank == 0: # Update progress bar only on rank 0
                log_msg_display = f'Loss: {total_loss.item():.4f}'
                if args.train_video_mae and video_loss_val is not None: log_msg_display += f', V_MAE: {(video_loss_val.item() if torch.is_tensor(video_loss_val) else video_loss_val):.4f}'
                if args.train_audio_mae and audio_loss_val is not None: log_msg_display += f', A_MAE: {(audio_loss_val.item() if torch.is_tensor(audio_loss_val) else audio_loss_val):.4f}'
                log_msg_display += f', LR: {current_lr:.6f}'
                progress_bar.set_postfix_str(log_msg_display)
                if step % args.log_interval == 0 and step > 0 : # More detailed periodic log from rank 0
                    tqdm.write(f"  Epoch {epoch+1}/{args.epochs}, Step {step}/{len(train_loader)} - {log_msg_display}")
                
        # --- End of Epoch --- 
        if num_batches == 0 and rank == 0: # Handle cases where a rank might process no batches if dataset is too small
            print(f"\nRank {rank} processed 0 batches in epoch {epoch+1}. Check dataset size and batch_size.")
            # If using DDP and drop_last=True, this should ideally not happen unless dataset is smaller than world_size * batch_size

        # Calculate average losses for the current rank
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_video_loss = epoch_video_loss / num_batches if num_batches > 0 and args.train_video_mae else 0
        avg_audio_loss = epoch_audio_loss / num_batches if num_batches > 0 and args.train_audio_mae else 0
        
        if is_ddp:
            # Optional: Gather losses from all ranks for a global average
            # This requires converting scalar losses to tensors and using all_reduce
            # For simplicity, we'll log rank 0's average.
            dist.barrier() # Ensure all ranks finish epoch computation

        if rank == 0: 
            progress_bar.close() # Close rank 0's progress bar
            print(f"\nEpoch {epoch+1} Finished.")
            print(f"  Average Loss (Rank 0): {avg_epoch_loss:.4f}")
            if args.train_video_mae: print(f"  Average Video MAE Loss (Rank 0): {avg_video_loss:.4f}")
            if args.train_audio_mae: print(f"  Average Audio MAE Loss (Rank 0): {avg_audio_loss:.4f}")

        # --- Checkpointing (only on master process) --- 
        if rank == 0 and ((epoch + 1) % args.checkpoint_interval == 0 or (epoch + 1) == args.epochs):
            checkpoint_path = os.path.join(args.output_dir, f't3_av_stage1_epoch_{epoch+1}.pt')
            # Save model.module.state_dict() when using DDP
            model_state_to_save = model.module.state_dict() if is_ddp else model.state_dict()
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model_state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': vars(args) # Store args used for this checkpoint
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        if is_ddp:
            dist.barrier() 

    if rank == 0: 
        print("Stage 1 MAE Training finished.")
        if hasattr(progress_bar, 'close') and rank == 0 : progress_bar.close() # Ensure it's closed if loop finishes early
    
    if is_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    # Need to import numpy for seeding
    import numpy as np 
    args = parse_args()
    main(args) 