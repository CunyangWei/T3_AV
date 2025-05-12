import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score # For potential more detailed metrics, though basic acc is calculated directly
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
    parser = argparse.ArgumentParser(description="Fine-tune T3-AV Model for Sound Classification")

    # --- Data Paths ---
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the vggsound.csv file.')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to the directory containing video files.')
    parser.add_argument('--output_dir', type=str, default='./finetune_output', help='Directory to save fine-tuning checkpoints and logs.')
    parser.add_argument('--pretrained_checkpoint', type=str, required=True, help='Path to the pretrained model checkpoint (Stage 1 or Stage 2).')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=30, help='Number of fine-tuning epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for fine-tuning.')
    parser.add_argument('--learning_rate', '--lr', type=float, default=5e-5, help='Peak learning rate for fine-tuning.') # Often lower for fine-tuning
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for AdamW optimizer.')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Number of warmup epochs.')

    # --- Model Configuration (These should ideally match the loaded checkpoint's architecture) ---
    # It's good practice to have these to reconstruct model arch if checkpoint doesn't save all args,
    # or for verification. The loaded checkpoint's state_dict is the source of truth for weights.
    parser.add_argument('--num_frames', type=int, default=16, help='Frames per video (must match pretrained model).')
    parser.add_argument('--vit_model_name', type=str, default='google/vit-base-patch16-224-in21k', help='ViT model (must match pretrained model).')
    parser.add_argument('--audio_duration', type=float, default=10.0, help='Audio duration (must match pretrained model).')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate (must match pretrained model).')
    parser.add_argument('--n_mels', type=int, default=128, help='Number of Mel bands (must match pretrained model).')
    parser.add_argument('--bb_layers', type=int, default=6, help='Shared backbone layers (must match pretrained model).')
    parser.add_argument('--bb_heads', type=int, default=12, help='Shared backbone heads (must match pretrained model).')
    parser.add_argument('--clf_hidden_dim', type=int, default=512, help='Hidden dimension for classification head MLP.')
    # Dummy args if T3_AV_Model __init__ needs them, even if not trained in finetuning (match defaults or pretrained)
    parser.add_argument('--dec_layers', type=int, default=2, help='Dummy MAE decoder layers for model init compatibility.')
    parser.add_argument('--dec_heads', type=int, default=8, help='Dummy MAE decoder heads for model init compatibility.')
    parser.add_argument('--proj_hidden_dim_contrastive', type=int, default=768, help='Dummy proj head hidden dim for model init compatibility.')
    parser.add_argument('--contrastive_dim', type=int, default=128, help='Dummy contrastive embed dim for model init compatibility.')

    # --- Fine-tuning Strategy ---
    parser.add_argument('--freeze_encoders', action='store_true', help='Freeze the video and audio modality encoders.')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze the shared transformer backbone.')

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
    # --- DDP Setup ---
    is_ddp = args.ddp
    rank = 0
    world_size = 1
    local_rank = 0

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

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    scaler = GradScaler() if args.amp and device.type == 'cuda' else None
    if scaler and rank == 0: print("AMP enabled, GradScaler initialized.")
    elif args.amp and device.type != 'cuda' and rank == 0:
        print("Warning: AMP requested but device is not CUDA. AMP will be disabled.")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory for fine-tuning: {args.output_dir}")
    if is_ddp: dist.barrier()

    # --- Dataset and DataLoader ---
    if rank == 0: print("Initializing Datasets and DataLoaders...")
    # Rank 0 creates a temporary train_dataset to get label_to_idx and num_classes
    label_to_idx_rank0 = None
    num_classes_rank0 = None
    if rank == 0:
        temp_train_dataset = VGGSoundDataset(csv_file_path=args.csv_path, video_dir=args.video_dir, split='train')
        if len(temp_train_dataset) == 0:
            print("Error: Training dataset is empty on Rank 0. Aborting.")
            if is_ddp: dist.destroy_process_group()
            return
        label_to_idx_rank0 = temp_train_dataset.label_to_idx
        num_classes_rank0 = temp_train_dataset.num_classes
        del temp_train_dataset

    if is_ddp:
        # Broadcast label_to_idx and num_classes from rank 0
        obj_list_label = [label_to_idx_rank0 if rank == 0 else None]
        dist.broadcast_object_list(obj_list_label, src=0)
        label_to_idx = obj_list_label[0]

        obj_list_num_classes = [num_classes_rank0 if rank == 0 else None] # Store as list for broadcast
        dist.broadcast_object_list(obj_list_num_classes, src=0)
        num_classes = obj_list_num_classes[0]
    else:
        label_to_idx = label_to_idx_rank0
        num_classes = num_classes_rank0
    
    if num_classes is None or num_classes <= 0:
        print(f"Error: num_classes is invalid ({num_classes}). Aborting.")
        if is_ddp: dist.destroy_process_group()
        return

    train_dataset = VGGSoundDataset(csv_file_path=args.csv_path, video_dir=args.video_dir, split='train', label_to_idx=label_to_idx)
    val_dataset = VGGSoundDataset(csv_file_path=args.csv_path, video_dir=args.video_dir, split='test', label_to_idx=label_to_idx)

    if len(train_dataset) == 0 and rank == 0: # Should have been caught by rank0 check, but as safeguard
        print("Error: Training dataset is empty after DDP sync.")
        if is_ddp: dist.destroy_process_group()
        return
    if len(val_dataset) == 0 and rank == 0:
        print("Warning: Validation dataset is empty.")

    if rank == 0: print(f"Number of classes for fine-tuning: {num_classes}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if is_ddp else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        collate_fn=vggsound_collate_fn, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'), sampler=train_sampler, drop_last=True if is_ddp else False
    )
    val_loader = None
    if len(val_dataset) > 0:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=vggsound_collate_fn, num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'), sampler=val_sampler, drop_last=False
        )
    if rank == 0: print("Datasets and DataLoaders initialized.")

    # --- Model Initialization ---
    if rank == 0: print("Initializing T3_AV_Model for Fine-tuning...")
    video_modality_params = {'num_frames_to_sample': args.num_frames, 'pretrained_vit_name': args.vit_model_name, 'output_embed_dim': 768} # Assuming 768 for ViT base
    audio_modality_params = {
        'sample_rate': args.sample_rate, 'n_mels': args.n_mels, 'target_spectrogram_height': 224, 'target_spectrogram_width': 224,
        'pretrained_vit_name': args.vit_model_name, 'audio_duration_secs': args.audio_duration,
        'n_fft': 1024, 'hop_length': 512, 'output_embed_dim': 768 # Assuming 768 for ViT base
    }
    shared_backbone_params = {'num_heads': args.bb_heads, 'num_layers': args.bb_layers, 'mlp_dim': 768 * 4, 'dropout': 0.1}
    mae_decoder_base_params = {'num_heads': args.dec_heads, 'num_decoder_layers': args.dec_layers, 'mlp_dim': 768 * 4, 'dropout': 0.1}
    proj_head_base_params = {'hidden_dim': args.proj_hidden_dim_contrastive, 'use_gelu': True}
    sound_clf_head_params = {'hidden_dim': args.clf_hidden_dim, 'num_classes': num_classes, 'use_gelu': True}

    model = T3_AV_Model(
        video_modality_params=video_modality_params, audio_modality_params=audio_modality_params,
        shared_backbone_params=shared_backbone_params, video_mae_decoder_base_params=mae_decoder_base_params,
        audio_mae_decoder_base_params=mae_decoder_base_params, proj_head_base_params=proj_head_base_params,
        sound_clf_head_params=sound_clf_head_params, contrastive_embed_dim=args.contrastive_dim,
        mae_mask_ratio=0.75 # Default, not used in finetune forward
    )
    if rank == 0: print("Model structure initialized.")

    # --- Load Pretrained Checkpoint ---
    if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
        if rank == 0: print(f"Loading pretrained checkpoint: {args.pretrained_checkpoint}")
        # Load on CPU first to avoid GPU memory issues on rank 0 if checkpoint is large
        checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu')
        raw_checkpoint_state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Clean keys: remove '_orig_mod.' prefix if present (from torch.compile)
        # and 'module.' prefix if present (from DDP)
        cleaned_state_dict = {}
        keys_to_skip_loading = [] # For logging keys we intentionally skip

        for k, v in raw_checkpoint_state_dict.items():
            new_k = k
            if new_k.startswith('_orig_mod.'):
                new_k = new_k[len('_orig_mod.'):]
            if new_k.startswith('module.'):
                new_k = new_k[len('module.'):]
            
            # For fine-tuning, always skip loading the sound_classifier weights
            # as it will be re-initialized for the target number of classes.
            if new_k.startswith('sound_classifier.'):
                keys_to_skip_loading.append(k) # Log original key name
                continue # Skip adding this key to cleaned_state_dict
                
            cleaned_state_dict[new_k] = v
        
        if rank == 0 and keys_to_skip_loading:
            print(f"Intentionally_skipped_loading_these_keys_from_checkpoint: {keys_to_skip_loading}")

        # Filter out keys that should not be loaded or might cause issues
        # e.g., MAE decoders, contrastive projection heads, optimizer state
        # The sound_classifier head might be different if num_classes has changed.
        # `load_state_dict` with `strict=False` will handle missing/unexpected keys.
        # If the sound_classifier in the checkpoint has a different num_classes, its weights won't load due to size mismatch.
        # This is generally the desired behavior (reinitialize classification head for new num_classes).
        
        # Example of keys that might be in a Stage 1/2 checkpoint but not needed/different for finetuning:
        # 'video_mae_decoder.*', 'audio_mae_decoder.*',
        # 'video_projection_head.*', 'audio_projection_head.*'
        # If 'sound_classifier.*' exists and num_classes matches, it will be loaded. Otherwise, it's ignored (good).

        load_result = model.load_state_dict(cleaned_state_dict, strict=False)
        if rank == 0:
            print(f"Loaded pretrained checkpoint into model (after key cleaning).")
            if load_result.missing_keys:
                print("Missing keys (expected for new/modified heads, or if checkpoint was partial):")
                for key in load_result.missing_keys: print(f"  {key}")
            if load_result.unexpected_keys:
                print("Unexpected keys (present in checkpoint but not in current model structure):")
                for key in load_result.unexpected_keys: print(f"  {key}")
    else:
        if rank == 0: print("Warning: No pretrained checkpoint provided or path does not exist. Training from scratch (ensure this is intended for finetuning script).")

    model = model.to(device)

    # --- torch.compile() ---
    if args.compile_model:
        if rank == 0: print(f"Compiling model with torch.compile(mode='{args.compile_mode}')...")
        try:
            model = torch.compile(model, mode=args.compile_mode)
            if rank == 0: print("Model compiled successfully.")
        except Exception as e:
            if rank == 0: print(f"Warning: torch.compile() failed: {e}. Proceeding without.")
    
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) # Can be False for finetuning if sure
        if rank == 0: print("Model wrapped with DDP.")

    # --- Freezing Components ---
    model_to_configure = model.module if is_ddp else model
    if args.freeze_encoders:
        if rank == 0: print("Freezing Modality Encoders (Video & Audio)...")
        for param in model_to_configure.video_encoder.parameters(): param.requires_grad = False
        for param in model_to_configure.audio_encoder.parameters(): param.requires_grad = False
    if args.freeze_backbone:
        if rank == 0: print("Freezing Shared Transformer Backbone...")
        for param in model_to_configure.shared_backbone.parameters(): param.requires_grad = False
            
    if rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total.")
        if trainable_params == 0 and args.epochs > 0:
            print("Warning: No parameters are trainable. Check freezing flags or model structure.")

    # --- Optimizer & Scheduler ---
    trainable_model_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_model_params and args.epochs > 0:
        if rank == 0: print("Error: No trainable parameters found for the optimizer, but epochs > 0.")
        if is_ddp: dist.destroy_process_group()
        return

    optimizer = optim.AdamW(trainable_model_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    if rank == 0: print("Optimizer and Scheduler initialized.")

    # --- Save Configuration ---
    if rank == 0:
        config_path = os.path.join(args.output_dir, 'finetune_config.json')
        with open(config_path, 'w') as f:
            args_dict = vars(args)
            args_dict['loaded_pretrained_checkpoint'] = args.pretrained_checkpoint
            args_dict['num_classes'] = num_classes
            json.dump(args_dict, f, indent=4)
        print(f"Saved fine-tuning configuration to {config_path}")

    # --- Training & Evaluation Loop ---
    if rank == 0:
        print("Starting Fine-tuning for Sound Classification...")
        best_val_accuracy = 0.0

    for epoch in range(args.epochs):
        if is_ddp: train_sampler.set_epoch(epoch)

        model.train()
        epoch_train_loss_sum = torch.tensor(0.0, device=device)
        processed_samples_in_train_epoch_sum = torch.tensor(0, device=device)
        correct_train_predictions_sum = torch.tensor(0, device=device)
        
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]", leave=(rank==0), disable=(rank!=0))
        for step, batch in enumerate(progress_bar_train):
            video_paths = batch['video_paths_batch']
            labels = batch['labels_batch'].to(device)
            if not video_paths: continue

            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
                try:
                    loss, logits, processed_in_step = (model.module if is_ddp else model).forward_finetune(video_paths, labels, device)
                except Exception as e:
                    print(f"Rank {rank}, Epoch {epoch+1}, Step {step}: Error during train forward: {e}. Paths: {video_paths}. Skipping.")
                    continue

                if loss is None or not torch.isfinite(loss) or processed_in_step == 0:
                    if processed_in_step == 0 and (loss is None or (hasattr(loss, 'item') and loss.item() == 0.0)): pass # Valid case if batch was entirely unprocessable
                    else: print(f"Rank {rank}: Non-finite train loss ({loss.item() if hasattr(loss, 'item') else 'None'}) or zero processed. Skipping.")
                    continue
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            scheduler.step()

            epoch_train_loss_sum += loss.item() * processed_in_step
            processed_samples_in_train_epoch_sum += processed_in_step
            correct_train_predictions_sum += (torch.argmax(logits, dim=1) == labels[:processed_in_step]).sum().item()
            
            if rank == 0:
                progress_bar_train.set_postfix_str(f'Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.7f}')
        
        if is_ddp:
            dist.all_reduce(epoch_train_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(processed_samples_in_train_epoch_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_train_predictions_sum, op=dist.ReduceOp.SUM)

        avg_train_loss = epoch_train_loss_sum.item() / processed_samples_in_train_epoch_sum.item() if processed_samples_in_train_epoch_sum.item() > 0 else 0
        train_accuracy = correct_train_predictions_sum.item() / processed_samples_in_train_epoch_sum.item() if processed_samples_in_train_epoch_sum.item() > 0 else 0
        
        if rank == 0:
            print(f"Epoch {epoch+1} Train: Avg Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f} (Total Proc: {processed_samples_in_train_epoch_sum.item()})")
            if hasattr(progress_bar_train, 'close'): progress_bar_train.close()

        # --- Validation ---
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
                    
                    with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
                        try:
                            loss, logits, processed_in_step = (model.module if is_ddp else model).forward_finetune(video_paths, labels, device)
                        except Exception as e:
                            print(f"Rank {rank}, Epoch {epoch+1}, Val Step {batch_idx}: Error during val forward: {e}. Skipping.")
                            continue
                        
                        if loss is None or not torch.isfinite(loss) or processed_in_step == 0:
                            if processed_in_step == 0 and (loss is None or (hasattr(loss, 'item') and loss.item() == 0.0)): pass
                            else: print(f"Rank {rank}: Non-finite val loss ({loss.item() if hasattr(loss, 'item') else 'None'}) or zero processed. Skipping.")
                            continue

                    epoch_val_loss_sum += loss.item() * processed_in_step
                    processed_samples_in_val_epoch_sum += processed_in_step
                    correct_val_predictions_sum += (torch.argmax(logits, dim=1) == labels[:processed_in_step]).sum().item()
            
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
                    best_model_path = os.path.join(args.output_dir, 't3_av_finetune_best_model.pt')
                    torch.save({
                        'epoch': epoch + 1, 'model_state_dict': (model.module if is_ddp else model).state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'args': vars(args),
                        'val_accuracy': val_accuracy, 'train_accuracy': train_accuracy # train_accuracy is from current epoch
                    }, best_model_path)
                    print(f"New best model saved to {best_model_path} with Val Acc: {val_accuracy:.4f}")

        # --- Save Epoch Checkpoint ---
        if rank == 0 and ((epoch + 1) % args.checkpoint_save_interval == 0 or (epoch + 1) == args.epochs):
            checkpoint_path = os.path.join(args.output_dir, f't3_av_finetune_epoch_{epoch+1}.pt')
            save_obj = {
                'epoch': epoch + 1, 'model_state_dict': (model.module if is_ddp else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'args': vars(args),
                'train_accuracy': train_accuracy # From current epoch
            }
            if val_loader and processed_samples_in_val_epoch_sum.item() > 0: # Add val_accuracy if computed
                 save_obj['val_accuracy'] = val_accuracy
            torch.save(save_obj, checkpoint_path)
            print(f"Fine-tuning checkpoint saved to {checkpoint_path}")
        
        if is_ddp: dist.barrier()

    if rank == 0:
        print("Fine-tuning finished.")
        print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
        if hasattr(progress_bar_train, 'close'): progress_bar_train.close()
        if val_loader and hasattr(progress_bar_val, 'close'): progress_bar_val.close()

    if is_ddp: dist.destroy_process_group()

if __name__ == '__main__':
    args = parse_args()
    main(args) 