import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import get_cosine_schedule_with_warmup

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# AMP imports
from torch.cuda.amp import GradScaler

from t3_av_model import T3_AV_Model
from vggsound_dataset import VGGSoundDataset, vggsound_collate_fn

def parse_args():
    parser = argparse.ArgumentParser(description="Train T3-AV Model for Sound Classification FROM SCRATCH")

    # --- Data Paths ---
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the vggsound.csv file.')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to the directory containing video files.')
    parser.add_argument('--output_dir', type=str, default='./from_scratch_output', help='Directory to save from-scratch training checkpoints and logs.')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.') # May need more than finetuning
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-4, help='Peak learning rate.') # May be higher than finetuning
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for AdamW optimizer.')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs.')

    # --- Model Configuration (Needed for T3_AV_Model instantiation) ---
    parser.add_argument('--num_frames', type=int, default=16, help='Frames per video.')
    parser.add_argument('--vit_model_name', type=str, default='google/vit-base-patch16-224-in21k', help='ViT model for encoders.')
    parser.add_argument('--audio_duration', type=float, default=10.0, help='Audio duration.')
    parser.add_argument('--bb_layers', type=int, default=6, help='Shared backbone layers.')
    parser.add_argument('--bb_heads', type=int, default=12, help='Shared backbone heads.')
    parser.add_argument('--clf_hidden_dim', type=int, default=512, help='Hidden dimension for classification head MLP.')
    parser.add_argument('--dec_layers', type=int, default=2, help='Dummy MAE decoder layers for model init.')
    parser.add_argument('--dec_heads', type=int, default=8, help='Dummy MAE decoder heads for model init.')
    parser.add_argument('--proj_hidden_dim_contrastive', type=int, default=768, help='Dummy proj head hidden dim for model init.')
    parser.add_argument('--contrastive_dim', type=int, default=128, help='Dummy contrastive embed dim for model init.')

    # --- System Configuration ---
    parser.add_argument('--device', type=str, default='cuda', help="Device to use ('cuda' or 'cpu').")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for DataLoader.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--log_interval', type=int, default=50, help='Log training status every N steps.')
    parser.add_argument('--checkpoint_save_interval', type=int, default=1, help='Save checkpoint every N epochs.')
    parser.add_argument('--eval_interval', type=int, default=1, help='Evaluate on validation set every N epochs.')

    # --- DDP Configuration ---
    parser.add_argument('--ddp', action='store_true', help='Enable Distributed Data Parallel training.')

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
            # Fallback for non-torchrun launches if SLURM variables are set
            if "SLURM_PROCID" in os.environ:
                os.environ["RANK"] = os.environ["SLURM_PROCID"]
                os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
                os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
                if "MASTER_ADDR" not in os.environ and "SLURM_LAUNCH_NODE_IPADDR" in os.environ:
                     os.environ["MASTER_ADDR"] = os.environ["SLURM_LAUNCH_NODE_IPADDR"]
                if "MASTER_PORT" not in os.environ:
                     os.environ["MASTER_PORT"] = "29500" # Default port
            else:
                raise RuntimeError("DDP enabled but LOCAL_RANK not set. Use torchrun or ensure SLURM env vars for DDP are set.")

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
        print(f"Output directory for from-scratch training: {args.output_dir}")
    
    if is_ddp:
        dist.barrier() # Ensure output dir is created by rank 0

    # --- Initialize Dataset & DataLoader ---
    if rank == 0: print("Initializing Datasets and DataLoaders...")
    train_dataset = VGGSoundDataset(csv_file_path=args.csv_path, video_dir=args.video_dir, split='train')
    if len(train_dataset) == 0:
        if rank == 0: print("Error: Training dataset is empty.")
        if is_ddp: dist.destroy_process_group()
        return
        
    # Create label_to_idx on rank 0 and broadcast it
    label_to_idx = None
    if rank == 0:
        label_to_idx = train_dataset.label_to_idx
    
    if is_ddp:
        # Broadcast label_to_idx from rank 0 to all other processes
        # PyTorch's broadcast_object_list can be used for this
        object_list = [label_to_idx if rank == 0 else None]
        dist.broadcast_object_list(object_list, src=0)
        label_to_idx = object_list[0]
        # All ranks need to re-initialize dataset with the same label_to_idx for val_dataset
        # Or ensure VGGSoundDataset handles label_to_idx=None correctly if it's only for mapping
        # For simplicity, let's assume train_dataset.label_to_idx is consistent if created once.
        # However, for val_dataset, it's crucial.
        if rank != 0: # Re-init train_dataset on other ranks if label_to_idx was critical for its internal state beyond num_classes
             train_dataset = VGGSoundDataset(csv_file_path=args.csv_path, video_dir=args.video_dir, split='train', label_to_idx=label_to_idx)


    val_dataset = VGGSoundDataset(csv_file_path=args.csv_path, video_dir=args.video_dir, split='test', label_to_idx=label_to_idx)
    if len(val_dataset) == 0 and rank == 0:
        print("Warning: Validation dataset is empty.")

    num_classes = train_dataset.num_classes # Should be consistent across ranks now
    if rank == 0: print(f"Number of classes: {num_classes}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if is_ddp else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        collate_fn=vggsound_collate_fn, num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        sampler=train_sampler, drop_last=True if is_ddp else False # drop_last for DDP
    )
    
    val_loader = None
    if len(val_dataset) > 0:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=vggsound_collate_fn, num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            sampler=val_sampler, drop_last=False # Usually not needed to drop last for validation
        )
    if rank == 0: print("Datasets and DataLoaders initialized.")
    
    video_modality_params = {'num_frames_to_sample': args.num_frames, 'pretrained_vit_name': args.vit_model_name, 'output_embed_dim': 768}
    audio_modality_params = {
        'sample_rate': 16000, 'n_mels': 128, 'target_spectrogram_height': 224, 'target_spectrogram_width': 224,
        'pretrained_vit_name': args.vit_model_name, 'audio_duration_secs': args.audio_duration,
        'n_fft': 1024, 'hop_length': 512, 'output_embed_dim': 768
    }
    # Ensure these MLP dims match the expectations of T3_AV_Model for consistency if MAE/Proj parts were ever used
    # For "from scratch" only the audio/video encoders, shared_backbone, and sound_classifier are truly trained.
    shared_backbone_params = {'num_heads': args.bb_heads, 'num_layers': args.bb_layers, 'mlp_dim': 768 * 4, 'dropout': 0.1} 
    mae_decoder_base_params = {'num_heads': args.dec_heads, 'num_decoder_layers': args.dec_layers, 'mlp_dim': 768 * 4, 'dropout': 0.1}
    proj_head_base_params = {'hidden_dim': args.proj_hidden_dim_contrastive, 'use_gelu': True}
    sound_clf_head_params = {'hidden_dim': args.clf_hidden_dim, 'num_classes': num_classes, 'use_gelu': True}

    if rank == 0: print("Initializing T3_AV_Model FROM SCRATCH...")
    model = T3_AV_Model(
        video_modality_params=video_modality_params,
        audio_modality_params=audio_modality_params,
        shared_backbone_params=shared_backbone_params,
        video_mae_decoder_base_params=mae_decoder_base_params,
        audio_mae_decoder_base_params=mae_decoder_base_params,
        proj_head_base_params=proj_head_base_params,
        sound_clf_head_params=sound_clf_head_params,
        contrastive_embed_dim=args.contrastive_dim,
        mae_mask_ratio=0.75 
    ) # No .to(device) here yet, DDP will handle it or compile will.
    if rank == 0: print("Model initialized with random weights (except ViT encoders if they load their own).")

    model = model.to(device) # Move model to device before compile or DDP

    # --- torch.compile() ---
    if args.compile_model:
        if rank == 0: print(f"Compiling model with torch.compile(mode='{args.compile_mode}')...")
        try:
            model = torch.compile(model, mode=args.compile_mode)
            if rank == 0: print("Model compiled successfully.")
        except Exception as e:
            if rank == 0: print(f"Warning: torch.compile() failed with error: {e}. Proceeding without compilation.")
    
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) # find_unused_parameters can be False if sure
        if rank == 0: print("Model wrapped with DDP.")
    
    if rank == 0:
        num_trainable_params = sum(p.numel() for p in (model.module if is_ddp else model).parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in (model.module if is_ddp else model).parameters())
        print(f"Model parameters: {num_trainable_params:,} trainable / {num_total_params:,} total.")

    optimizer = optim.AdamW((model.module if is_ddp else model).parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    print("Optimizer and Scheduler initialized.")

    if rank == 0:
        config_path = os.path.join(args.output_dir, 'from_scratch_config.json')
        with open(config_path, 'w') as f:
            args_dict = vars(args)
            args_dict['training_type'] = 'from_scratch'
            args_dict['num_classes'] = num_classes
            json.dump(args_dict, f, indent=4)
        print(f"Saved from-scratch configuration to {config_path}")

    if rank == 0:
        print("Starting Training FROM SCRATCH for Sound Classification...")
        best_val_accuracy = 0.0

    for epoch in range(args.epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch) # Important for shuffling with DDP

        model.train()
        epoch_train_loss_sum = torch.tensor(0.0, device=device) # For DDP aggregation
        processed_samples_in_train_epoch_sum = torch.tensor(0, device=device)
        correct_train_predictions_sum = torch.tensor(0, device=device)
        
        # Use rank for tqdm disable
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]", leave=(rank==0), disable=(rank!=0))
        for step, batch in enumerate(progress_bar_train):
            video_paths = batch['video_paths_batch']
            labels = batch['labels_batch'].to(device)
            if not video_paths: continue

            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                try:
                    # Access .module if DDP
                    loss, logits, processed_batch_size_in_step = (model.module if is_ddp else model).forward_finetune(video_paths, labels, device)
                except Exception as e:
                    print(f"Rank {rank}: Error during train forward pass: {e}. Paths: {video_paths}. Skipping batch.")
                    continue

                if loss is None or not torch.isfinite(loss) or processed_batch_size_in_step == 0:
                    if processed_batch_size_in_step == 0 and (loss is None or (hasattr(loss, 'item') and loss.item() == 0.0)): # Check if loss is tensor
                        pass
                    else:
                        print(f"Rank {rank}: Non-finite train loss ({loss.item() if hasattr(loss, 'item') else 'None'}) or zero processed samples. Skipping.")
                    continue
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            scheduler.step()

            epoch_train_loss_sum += loss.item() * processed_batch_size_in_step 
            processed_samples_in_train_epoch_sum += processed_batch_size_in_step
            correct_train_predictions_sum += (torch.argmax(logits, dim=1) == labels[:processed_batch_size_in_step]).sum().item()
            
            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                # Display instantaneous loss for rank 0's current batch
                progress_bar_train.set_postfix_str(f'Loss: {loss.item():.4f}, LR: {current_lr:.7f}')
        
        if is_ddp:
            dist.all_reduce(epoch_train_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(processed_samples_in_train_epoch_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_train_predictions_sum, op=dist.ReduceOp.SUM)

        avg_train_loss = epoch_train_loss_sum.item() / processed_samples_in_train_epoch_sum.item() if processed_samples_in_train_epoch_sum.item() > 0 else 0
        train_accuracy = correct_train_predictions_sum.item() / processed_samples_in_train_epoch_sum.item() if processed_samples_in_train_epoch_sum.item() > 0 else 0
        
        if rank == 0:
            print(f"Epoch {epoch+1} Train: Avg Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f} (Total Proc: {processed_samples_in_train_epoch_sum.item()})")
            if hasattr(progress_bar_train, 'close'): progress_bar_train.close()


        if val_loader and (epoch + 1) % args.eval_interval == 0:
            model.eval()
            epoch_val_loss_sum = torch.tensor(0.0, device=device)
            processed_samples_in_val_epoch_sum = torch.tensor(0, device=device)
            correct_val_predictions_sum = torch.tensor(0, device=device)
            
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [VAL]", leave=(rank==0), disable=(rank!=0))
            with torch.no_grad():
                for batch_idx, batch in enumerate(progress_bar_val):
                    video_paths = batch['video_paths_batch']
                    labels = batch['labels_batch'].to(device)
                    if not video_paths: continue
                    
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        try:
                            loss, logits, processed_batch_size_in_step = (model.module if is_ddp else model).forward_finetune(video_paths, labels, device)
                        except Exception as e:
                            print(f"Rank {rank}: Error during val forward pass: {e}. Paths: {video_paths}. Skipping batch.")
                            continue
                        
                        if loss is None or not torch.isfinite(loss) or processed_batch_size_in_step == 0:
                            if processed_batch_size_in_step == 0 and (loss is None or (hasattr(loss, 'item') and loss.item() == 0.0)):
                                pass
                            else:
                                print(f"Rank {rank}: Non-finite val loss ({loss.item() if hasattr(loss, 'item') else 'None'}) or zero processed samples. Skipping.")
                            continue

                    epoch_val_loss_sum += loss.item() * processed_batch_size_in_step
                    processed_samples_in_val_epoch_sum += processed_batch_size_in_step
                    correct_val_predictions_sum += (torch.argmax(logits, dim=1) == labels[:processed_batch_size_in_step]).sum().item()
            
            if is_ddp:
                dist.all_reduce(epoch_val_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(processed_samples_in_val_epoch_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(correct_val_predictions_sum, op=dist.ReduceOp.SUM)

            avg_val_loss = epoch_val_loss_sum.item() / processed_samples_in_val_epoch_sum.item() if processed_samples_in_val_epoch_sum.item() > 0 else 0
            val_accuracy = correct_val_predictions_sum.item() / processed_samples_in_val_epoch_sum.item() if processed_samples_in_val_epoch_sum.item() > 0 else 0
            
            if rank == 0:
                print(f"Epoch {epoch+1} Val: Avg Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.4f} (Total Proc: {processed_samples_in_val_epoch_sum.item()})")
                if hasattr(progress_bar_val, 'close'): progress_bar_val.close()

                if val_accuracy > best_val_accuracy and processed_samples_in_val_epoch_sum.item() > 0:
                    best_val_accuracy = val_accuracy
                    best_model_path = os.path.join(args.output_dir, 't3_av_from_scratch_best_model.pt')
                    torch.save({
                        'epoch': epoch + 1, 'model_state_dict': (model.module if is_ddp else model).state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'args': vars(args),
                        'val_accuracy': val_accuracy, 'train_accuracy': train_accuracy
                    }, best_model_path)
                    print(f"New best model (from scratch) saved to {best_model_path} with Val Acc: {val_accuracy:.4f}")

        if rank == 0 and ((epoch + 1) % args.checkpoint_save_interval == 0 or (epoch + 1) == args.epochs):
            checkpoint_path = os.path.join(args.output_dir, f't3_av_from_scratch_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': (model.module if is_ddp else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy if val_loader and processed_samples_in_val_epoch_sum.item() > 0 else None, # val_accuracy is rank 0 specific
                'train_accuracy': train_accuracy, # train_accuracy is rank 0 specific (global)
                'args': vars(args)
            }, checkpoint_path)
            print(f"From-scratch checkpoint saved to {checkpoint_path}")
        
        if is_ddp:
            dist.barrier() # Ensure all processes are synced before next epoch / saving

    if rank == 0:
        print("FROM SCRATCH Sound Classification Training finished.")
        print(f"Best Validation Accuracy (from scratch): {best_val_accuracy:.4f}")
        # Ensure progress bars are closed if training finishes early or loop breaks
        if hasattr(progress_bar_train, 'close'): progress_bar_train.close()
        if val_loader and hasattr(progress_bar_val, 'close'): progress_bar_val.close()


    if is_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    args = parse_args()
    main(args) 