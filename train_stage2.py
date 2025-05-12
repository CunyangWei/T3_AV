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

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# AMP imports
from torch.cuda.amp import GradScaler

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

    # --- DDP Configuration ---
    parser.add_argument('--ddp', action='store_true', help='Enable Distributed Data Parallel training.')
    # local_rank is typically provided by the launch utility

    # --- AMP Configuration ---
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision training.')

    # --- Torch Compile Configuration ---
    parser.add_argument('--compile_model', action='store_true', help='Enable torch.compile() for the model.')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead', 
                        choices=['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
                        help='Mode for torch.compile().')

    args = parser.parse_args()
    return args

def main(args):
    # --- DDP Setup (if enabled) ---
    is_ddp = args.ddp
    rank = 0
    world_size = 1
    local_rank = 0 # Default for non-DDP or master process

    if is_ddp:
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("DDP enabled but LOCAL_RANK not set. Use torchrun or similar.")
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
    torch.manual_seed(args.seed + rank) # Add rank for different seeds per process
    np.random.seed(args.seed + rank)
    
    # AMP: Initialize GradScaler if AMP is enabled and device is CUDA
    scaler = None
    use_amp = args.amp and device.type == 'cuda'
    if use_amp:
        scaler = GradScaler()
        if rank == 0: print("AMP enabled, GradScaler initialized.")
    elif args.amp and device.type != 'cuda' and rank == 0:
        print("Warning: AMP requested but device is not CUDA. AMP will be disabled.")
        
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
    if is_ddp: dist.barrier() # Ensure output_dir is created before other ranks might try to write
    if rank == 0 or not is_ddp: print(f"Process Rank {rank}: Using device: {device}")


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
    if rank == 0: print("Initializing T3_AV_Model for Stage 2...")
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
    if rank == 0: print("Model structure initialized.")

    # --- Load Stage 1 Checkpoint --- 
    # Rank 0 loads the checkpoint, then DDP synchronizes the model state.
    if rank == 0:
        if not os.path.exists(args.stage1_checkpoint):
            print(f"Error: Stage 1 checkpoint not found at {args.stage1_checkpoint}")
            # Signal other processes to exit if checkpoint is missing
            # A simple way is to let them error out when model isn't loaded,
            # or use dist.broadcast to send a status flag.
            # For now, we'll let DDP model sync fail if rank 0 doesn't load.
            # A more robust solution would involve explicit signaling.
            dist.barrier() # Wait for other ranks before potentially erroring them
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {args.stage1_checkpoint}")

        print(f"Loading Stage 1 checkpoint from: {args.stage1_checkpoint}")
        checkpoint = torch.load(args.stage1_checkpoint, map_location='cpu')
        checkpoint_state_dict = checkpoint.get('model_state_dict', checkpoint)

        # --- CRITICAL CHANGE: Do NOT remove shared_backbone.pos_encoder.pe ---
        keys_to_remove = []
        # pe_key = 'shared_backbone.pos_encoder.pe'
        # if pe_key in checkpoint_state_dict:
        #     print(f"Warning: Explicitly removing {pe_key} from Stage 1 checkpoint for Stage 2 loading. Ensure this is intended.")
        #     keys_to_remove.append(pe_key)
        
        # Remove keys that might cause issues if architectures slightly differ
        # (e.g., MAE decoders not present or differently structured in a model aimed only at contrastive)
        # Or if certain parts are meant to be re-initialized.
        # Example:
        # for key_pattern in ['mae_video_decoder.', 'mae_audio_decoder.']:
        #     keys_to_remove.extend([k for k in checkpoint_state_dict if k.startswith(key_pattern)])

        # If Stage 1 had a classification head not used in Stage 2
        # for key_pattern in ['sound_classifier.']:
        #    keys_to_remove.extend([k for k in checkpoint_state_dict if k.startswith(key_pattern)])

        for key in keys_to_remove:
            if key in checkpoint_state_dict:
                del checkpoint_state_dict[key]
                if rank == 0: print(f"Removed key from Stage 1 checkpoint: {key}")
        
        try:
            load_result = model.load_state_dict(checkpoint_state_dict, strict=False)
            print(f"Checkpoint load result (Rank 0): {load_result}")
            if load_result.missing_keys: print("Missing keys (Rank 0):", load_result.missing_keys)
            if load_result.unexpected_keys: print("Warning: Unexpected keys (Rank 0):", load_result.unexpected_keys)
        except Exception as e:
            print(f"Error loading state dict on Rank 0: {e}")
            dist.barrier() # Ensure other ranks wait
            raise e # Re-raise error to stop execution
        print("Stage 1 weights loaded into model on Rank 0.")

    if is_ddp:
        dist.barrier() # Ensure rank 0 has loaded the model before other ranks proceed to DDP wrapping

    model = model.to(device) # Move model to its assigned device

    # --- torch.compile() ---
    if args.compile_model:
        if rank == 0: print(f"Compiling model with torch.compile(mode='{args.compile_mode}')...")
        try:
            # Compile the model before DDP wrapping
            # Note: Some backends/modes might have issues with DDP or specific model structures.
            # Start with "reduce-overhead" or "default". "max-autotune" can be slow initially.
            model = torch.compile(model, mode=args.compile_mode)
            if rank == 0: print("Model compiled successfully.")
        except Exception as e:
            if rank == 0: print(f"Warning: torch.compile() failed with error: {e}. Proceeding without compilation.")
            # Optionally, you might want to fall back or raise an error depending on requirements
    
    if is_ddp:
        # find_unused_parameters=True is important if parts of the model are frozen
        # or if torch.compile changes parameter usage patterns for DDP.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if rank == 0: print("Model wrapped with DDP.")


    # --- Freezing Components --- 
    # Access .module if DDP is used, otherwise direct access
    model_to_configure = model.module if is_ddp else model

    if args.freeze_encoders:
        if rank == 0: print("Freezing Modality Encoders (Video & Audio)...")
        for name, param in model_to_configure.video_encoder.named_parameters():
            param.requires_grad = False
        for name, param in model_to_configure.audio_encoder.named_parameters():
            param.requires_grad = False
            
    if args.freeze_backbone:
        if rank == 0: print("Freezing Shared Transformer Backbone...")
        for name, param in model_to_configure.shared_backbone.named_parameters():
            param.requires_grad = False
            
    # Verify which parameters are trainable (on rank 0 for logging)
    if rank == 0:
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_trainable_params:,} trainable / {num_total_params:,} total")
        if num_trainable_params == 0 and args.epochs > 0 : # Only warn if actually trying to train
            print("Warning: No parameters are set to trainable. Check freezing flags.")
            # Potentially signal other ranks to exit if no training is possible.

    if is_ddp: dist.barrier()


    # --- Initialize Dataset & DataLoader --- 
    if rank == 0: print("Initializing Dataset and DataLoader for Stage 2...")
    train_dataset = VGGSoundDataset(csv_file_path=args.csv_path, video_dir=args.video_dir, split='train')
    
    if len(train_dataset) == 0:
        if rank == 0: print("Error: Training dataset is empty.")
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
    # Filter parameters to only optimize those that require gradients
    # model.parameters() will correctly yield parameters from model.module if DDP wrapped
    trainable_params = [p for p in model.parameters() if p.requires_grad] 
    if not trainable_params and args.epochs > 0: # Check if actually training
        if rank == 0: print("Error: No trainable parameters found for the optimizer.")
        if is_ddp: dist.barrier()
        return # Exit if no parameters to train and epochs > 0
        
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    if rank == 0: print("Optimizer and Scheduler initialized for trainable parameters.")
    
    # --- Save Stage 2 Configuration --- 
    if rank == 0:
        config_path = os.path.join(args.output_dir, 'stage2_config.json')
        with open(config_path, 'w') as f:
            args_dict = vars(args)
            args_dict['loaded_stage1_checkpoint'] = args.stage1_checkpoint
            json.dump(args_dict, f, indent=4)
        print(f"Saved Stage 2 configuration to {config_path}")

    if is_ddp: dist.barrier()

    # --- Training Loop --- 
    if rank == 0: print(f"Starting Stage 2 Contrastive Training on {world_size} GPU(s)...")
    for epoch in range(args.epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        model.train() 
        epoch_contrastive_loss = 0.0
        processed_samples_in_epoch = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Rank {rank})", leave=False, disable=(rank != 0))
        
        for step, batch in enumerate(progress_bar):
            video_paths = batch['video_paths_batch']
            
            if not video_paths: continue

            optimizer.zero_grad()
            
            # AMP: autocast context
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                try:
                     # Forward pass for Stage 2
                     if is_ddp:
                         contrastive_loss, batch_size_processed = model.module.forward_stage2(
                             video_paths_batch=video_paths, 
                             device=device 
                         )
                     else:
                         contrastive_loss, batch_size_processed = model.forward_stage2(
                             video_paths_batch=video_paths, 
                             device=device
                         )
                except Exception as e:
                     print(f"\nError during forward pass stage 2 at step {step} in epoch {epoch+1} on Rank {rank}: {e}")
                     print(f"Video paths in failing batch: {video_paths}")
                     print("Skipping this batch.")
                     # Ensure scaler state is updated even on error if step was skipped within autocast
                     # No explicit scaler.update() here as it's tied to optimizer.step()
                     continue 
                     
                if contrastive_loss is None or not torch.isfinite(contrastive_loss) or batch_size_processed == 0:
                     print(f"\nWarning: Non-finite loss ({contrastive_loss}) or zero processed samples ({batch_size_processed}) at step {step} on Rank {rank}. Skipping batch.")
                     # No explicit scaler.update() here
                     continue 
                 
            # Backward pass & optimization
            if use_amp:
                scaler.scale(contrastive_loss).backward()
                # Optional: Unscale gradients before clipping if you clip gradients
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                contrastive_loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            # --- Logging --- 
            current_lr = optimizer.param_groups[0]['lr']
            loss_item = contrastive_loss.item() # This is already a scalar
            epoch_contrastive_loss += loss_item * batch_size_processed 
            processed_samples_in_epoch += batch_size_processed
            
            if rank == 0:
                log_msg = f'Contrastive Loss: {loss_item:.4f}, LR: {current_lr:.6f}'
                progress_bar.set_postfix_str(log_msg)
                if step % args.log_interval == 0 and step > 0:
                    tqdm.write(f"  Epoch {epoch+1}/{args.epochs}, Step {step}/{len(train_loader)} - {log_msg}")
                
        # --- End of Epoch --- 
        # For DDP, each rank has its own sum. For a global average, you'd need to all_reduce.
        # Here, we log rank 0's average, which is fine if data distribution is similar.
        avg_epoch_loss_rank_local = epoch_contrastive_loss / processed_samples_in_epoch if processed_samples_in_epoch > 0 else 0
        
        if is_ddp:
            # Gather total loss and total samples from all ranks for accurate average
            total_loss_tensor = torch.tensor([epoch_contrastive_loss], dtype=torch.float64, device=device)
            total_samples_tensor = torch.tensor([processed_samples_in_epoch], dtype=torch.float64, device=device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
            
            global_avg_epoch_loss = total_loss_tensor.item() / total_samples_tensor.item() if total_samples_tensor.item() > 0 else 0
            if rank == 0:
                progress_bar.close()
                print(f"\nEpoch {epoch+1} Finished.")
                print(f"  Average Contrastive Loss (Global): {global_avg_epoch_loss:.4f}")
                # print(f"  Rank 0 Local Avg Loss: {avg_epoch_loss_rank_local:.4f}") # For debugging if needed
        else: # Non-DDP
            if rank == 0: # Should always be rank 0 if not DDP
                progress_bar.close()
                print(f"\nEpoch {epoch+1} Finished. Average Contrastive Loss: {avg_epoch_loss_rank_local:.4f}")


        # --- Checkpointing (only on master process) --- 
        if rank == 0 and ((epoch + 1) % args.checkpoint_interval == 0 or (epoch + 1) == args.epochs):
            checkpoint_path = os.path.join(args.output_dir, f't3_av_stage2_epoch_{epoch+1}.pt')
            # Save model.module.state_dict() when using DDP
            model_state_to_save = model.module.state_dict() if is_ddp else model.state_dict()
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model_state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': vars(args) 
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        if is_ddp:
            dist.barrier() # Ensure checkpoint is saved before next epoch or finishing

    if rank == 0: 
        print("Stage 2 Contrastive Training finished.")
        if hasattr(progress_bar, 'close') and rank == 0: progress_bar.close()

    if is_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    import numpy as np 
    args = parse_args()
    main(args) 