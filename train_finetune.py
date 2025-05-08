import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import argparse
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score # For calculating accuracy
from transformers import get_cosine_schedule_with_warmup

# Import project modules
from t3_av_model import T3_AV_Model
from vggsound_dataset import VGGSoundDataset, vggsound_collate_fn

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Fine-tuning T3-AV Model for Sound Classification")

    # --- Data Paths ---
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the vggsound.csv file.')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to the directory containing video files.')
    parser.add_argument('--pretrained_checkpoint', type=str, required=True, help='Path to the Stage 1 or Stage 2 pretrained checkpoint (.pt file).')
    parser.add_argument('--output_dir', type=str, default='./finetune_output', help='Directory to save fine-tuning checkpoints and logs.')

    # --- Fine-tuning Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=30, help='Number of fine-tuning epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for fine-tuning.')
    parser.add_argument('--learning_rate', '--lr', type=float, default=5e-5, help='Peak learning rate for fine-tuning.')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for AdamW optimizer.')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Number of warmup epochs for learning rate scheduler.')

    # --- Model Freezing Options ---
    parser.add_argument('--freeze_all_except_head', action='store_true', help='Freeze all layers except the classification head.')
    parser.add_argument('--unfreeze_backbone_layers', type=int, default=0, help='Number of top layers to unfreeze in the shared backbone (0 means backbone frozen).')
    parser.add_argument('--unfreeze_encoder_layers', type=int, default=0, help='Number of top layers to unfreeze in audio encoder ViT (0 means audio encoder frozen). Not implemented yet.')


    # --- Model Configuration (some loaded from checkpoint, some needed for instantiation) ---
    # These should ideally align with the loaded checkpoint's architecture.
    # T3_AV_Model __init__ requires these to build the structure before loading weights.
    parser.add_argument('--num_frames', type=int, default=16, help='Frames per video (from pretraining).')
    parser.add_argument('--vit_model_name', type=str, default='google/vit-base-patch16-224-in21k', help='ViT model (from pretraining).')
    parser.add_argument('--audio_duration', type=float, default=10.0, help='Audio duration (from pretraining).')
    parser.add_argument('--bb_layers', type=int, default=6, help='Shared backbone layers (from pretraining).')
    parser.add_argument('--bb_heads', type=int, default=12, help='Shared backbone heads (from pretraining).')
    # Classifier head specific
    parser.add_argument('--clf_hidden_dim', type=int, default=512, help='Hidden dimension for the classification head MLP.')
    
    # Dummy params for MAE/Projection heads if T3_AV_Model requires them in __init__
    # Their weights won't be used if not part of the fine-tuning task.
    parser.add_argument('--dec_heads', type=int, default=8, help='Dummy MAE decoder heads.')
    parser.add_argument('--proj_hidden_dim_contrastive', type=int, default=768, help='Dummy projection head hidden dim for contrastive.')
    parser.add_argument('--contrastive_dim', type=int, default=128, help='Dummy contrastive embedding dim.')


    # --- System Configuration ---
    parser.add_argument('--device', type=str, default='cuda', help="Device to use ('cuda' or 'cpu').")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for DataLoader.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--log_interval', type=int, default=50, help='Log training status every N steps.')
    parser.add_argument('--checkpoint_save_interval', type=int, default=1, help='Save checkpoint every N epochs.') # More frequent for finetuning
    parser.add_argument('--eval_interval', type=int, default=1, help='Evaluate on validation set every N epochs.')

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

    # --- Initialize Dataset & DataLoader ---
    print("Initializing Datasets and DataLoaders...")
    # Important: Use the same label_to_idx for train and val
    train_dataset = VGGSoundDataset(csv_file_path=args.csv_path, video_dir=args.video_dir, split='train')
    if len(train_dataset) == 0:
        print("Error: Training dataset is empty. Please check paths and split name.")
        return
    
    val_dataset = VGGSoundDataset(csv_file_path=args.csv_path, video_dir=args.video_dir, split='test', label_to_idx=train_dataset.label_to_idx)
    if len(val_dataset) == 0:
        print("Warning: Validation dataset is empty. Evaluation will be skipped.")

    num_classes = train_dataset.num_classes
    print(f"Number of classes: {num_classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=vggsound_collate_fn, num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=vggsound_collate_fn, num_workers=args.num_workers,
            pin_memory=True if args.device == 'cuda' else False
        )
    print("Datasets and DataLoaders initialized.")

    # --- Model Configuration for Instantiation ---
    video_modality_params = {'num_frames_to_sample': args.num_frames, 'pretrained_vit_name': args.vit_model_name, 'output_embed_dim': 768}
    audio_modality_params = {
        'sample_rate': 16000, 'n_mels': 128, 'target_spectrogram_height': 224, 'target_spectrogram_width': 224,
        'pretrained_vit_name': args.vit_model_name, 'audio_duration_secs': args.audio_duration,
        'n_fft': 1024, 'hop_length': 512, 'output_embed_dim': 768
    }
    shared_backbone_params = {'num_heads': args.bb_heads, 'num_layers': args.bb_layers, 'mlp_dim': 768 * 4, 'dropout': 0.1}
    mae_decoder_base_params = {'num_heads': args.dec_heads, 'num_decoder_layers': 1, 'mlp_dim': 768 * 2, 'dropout': 0.1} # Dummy
    proj_head_base_params = {'hidden_dim': args.proj_hidden_dim_contrastive, 'use_gelu': True} # Dummy for contrastive head
    sound_clf_head_params = {'hidden_dim': args.clf_hidden_dim, 'num_classes': num_classes, 'use_gelu': True}

    # --- Initialize Model ---
    print("Initializing T3_AV_Model for Fine-tuning...")
    model = T3_AV_Model(
        video_modality_params=video_modality_params,
        audio_modality_params=audio_modality_params,
        shared_backbone_params=shared_backbone_params,
        video_mae_decoder_base_params=mae_decoder_base_params,
        audio_mae_decoder_base_params=mae_decoder_base_params,
        proj_head_base_params=proj_head_base_params,
        sound_clf_head_params=sound_clf_head_params,
        contrastive_embed_dim=args.contrastive_dim, # Dummy
        mae_mask_ratio=0.75 # Dummy
    )
    print("Model structure initialized.")

    # --- Load Pretrained Checkpoint ---
    if not os.path.exists(args.pretrained_checkpoint):
        print(f"Error: Pretrained checkpoint not found at {args.pretrained_checkpoint}")
        return
    print(f"Loading pretrained checkpoint from: {args.pretrained_checkpoint}")
    checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu')
    
    try:
        # strict=False because the sound_classifier head is new and won't be in Stage1/2 checkpoints.
        # Also, MAE decoders or projection heads might be in the checkpoint but not used here.
        load_result = model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
        print(f"Checkpoint load result: Missing keys: {load_result.missing_keys}, Unexpected keys: {load_result.unexpected_keys}")
        if any('sound_classifier' not in key for key in load_result.missing_keys):
             print("Warning: Some missing keys are not from the sound_classifier. Check model architecture alignment.")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return
    model = model.to(device)
    print("Pretrained weights loaded into model.")

    # --- Freezing Components ---
    if args.freeze_all_except_head:
        print("Freezing all layers except the classification head.")
        for name, param in model.named_parameters():
            if 'sound_classifier' not in name:
                param.requires_grad = False
    else:
        # Freeze modality encoders by default if not unfreezing specific layers
        print("Freezing video and audio encoders by default (unless unfreeze_encoder_layers is used).")
        for param in model.video_encoder.parameters(): param.requires_grad = False
        for param in model.audio_encoder.parameters(): param.requires_grad = False # Audio encoder initially frozen

        # Unfreeze top N layers of audio encoder's ViT
        if args.unfreeze_encoder_layers > 0 and hasattr(model.audio_encoder, 'vit_model'):
            print(f"Unfreezing top {args.unfreeze_encoder_layers} layers of the audio encoder ViT.")
            # The ViT's encoder has a 'layer' attribute which is a ModuleList
            total_encoder_layers = len(model.audio_encoder.vit_model.encoder.layer)
            for i, layer in enumerate(model.audio_encoder.vit_model.encoder.layer):
                if i >= total_encoder_layers - args.unfreeze_encoder_layers:
                    print(f"  Unfreezing audio ViT encoder layer {i}")
                    for param in layer.parameters():
                        param.requires_grad = True
            # Also unfreeze the ViT's layernorm after the encoder and pooler if they exist and are used
            if hasattr(model.audio_encoder.vit_model, 'layernorm'):
                 for param in model.audio_encoder.vit_model.layernorm.parameters(): param.requires_grad = True
            # Pooler might not be directly used as we take CLS from backbone, but unfreeze if policy is to unfreeze "top" parts
            if hasattr(model.audio_encoder.vit_model, 'pooler'):
                 for param in model.audio_encoder.vit_model.pooler.parameters(): param.requires_grad = True


        # Freeze shared backbone by default, then unfreeze top N layers
        print("Freezing shared backbone by default (unless unfreeze_backbone_layers is used).")
        for param in model.shared_backbone.parameters(): param.requires_grad = False
        if args.unfreeze_backbone_layers > 0 and hasattr(model.shared_backbone, 'transformer_encoder'):
            print(f"Unfreezing top {args.unfreeze_backbone_layers} layers of the shared backbone.")
            total_backbone_layers = len(model.shared_backbone.transformer_encoder.layers)
            for i, layer in enumerate(model.shared_backbone.transformer_encoder.layers):
                if i >= total_backbone_layers - args.unfreeze_backbone_layers:
                    print(f"  Unfreezing shared backbone layer {i}")
                    for param in layer.parameters():
                        param.requires_grad = True
            # Also unfreeze the final norm of the backbone if it exists
            if hasattr(model.shared_backbone.transformer_encoder, 'norm') and model.shared_backbone.transformer_encoder.norm is not None:
                 print("  Unfreezing final norm of shared backbone.")
                 for param in model.shared_backbone.transformer_encoder.norm.parameters(): param.requires_grad = True
        
        # Classification head is always trainable
        for param in model.sound_classifier.parameters():
            param.requires_grad = True


    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_trainable_params:,} trainable / {num_total_params:,} total.")
    if num_trainable_params == 0:
        print("Warning: No parameters are set to trainable. Check freezing flags.")
        # return # Exit if nothing to train

    # --- Initialize Optimizer & Scheduler ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print("Error: No trainable parameters found for the optimizer.")
        return
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    print("Optimizer and Scheduler initialized for trainable parameters.")

    # --- Save Configuration ---
    config_path = os.path.join(args.output_dir, 'finetune_config.json')
    with open(config_path, 'w') as f:
        args_dict = vars(args)
        args_dict['loaded_pretrained_checkpoint'] = args.pretrained_checkpoint
        args_dict['num_classes'] = num_classes
        json.dump(args_dict, f, indent=4)
    print(f"Saved fine-tuning configuration to {config_path}")

    # --- Training & Validation Loop ---
    print("Starting Fine-tuning for Sound Classification...")
    best_val_accuracy = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0.0
        processed_samples_in_train_epoch = 0
        correct_train_predictions = 0
        
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [TRAIN]", leave=True)
        for step, batch in enumerate(progress_bar_train):
            video_paths = batch['video_paths_batch']
            labels = batch['labels_batch'].to(device)
            if not video_paths: continue

            optimizer.zero_grad()
            try:
                loss, logits, processed_batch_size_in_step = model.forward_finetune(video_paths, labels, device)
            except Exception as e:
                print(f"Error during train forward pass: {e}. Paths: {video_paths}. Skipping batch.")
                continue

            if loss is None or not torch.isfinite(loss) or processed_batch_size_in_step == 0:
                if processed_batch_size_in_step == 0 and (loss is None or loss.item() == 0.0):
                    pass
                else:
                    print(f"Non-finite train loss ({loss.item() if loss is not None else 'None'}) or zero processed samples. Skipping batch.")
                continue
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item() * processed_batch_size_in_step 
            processed_samples_in_train_epoch += processed_batch_size_in_step
            correct_train_predictions += (torch.argmax(logits, dim=1) == labels[:processed_batch_size_in_step]).sum().item()
            
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar_train.set_postfix_str(f'Loss: {loss.item():.4f}, LR: {current_lr:.7f}')

        avg_train_loss = epoch_train_loss / processed_samples_in_train_epoch if processed_samples_in_train_epoch > 0 else 0
        train_accuracy = correct_train_predictions / processed_samples_in_train_epoch if processed_samples_in_train_epoch > 0 else 0
        print(f"Epoch {epoch+1} Train: Avg Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f} (Processed: {processed_samples_in_train_epoch})")

        # --- Validation ---
        if val_loader and (epoch + 1) % args.eval_interval == 0:
            model.eval()
            epoch_val_loss = 0.0
            processed_samples_in_val_epoch = 0
            correct_val_predictions = 0
            
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [VAL]", leave=True)
            with torch.no_grad():
                for batch in progress_bar_val:
                    video_paths = batch['video_paths_batch']
                    labels = batch['labels_batch'].to(device)
                    if not video_paths: continue
                    
                    try:
                        loss, logits, processed_batch_size_in_step = model.forward_finetune(video_paths, labels, device)
                    except Exception as e:
                        print(f"Error during val forward pass: {e}. Paths: {video_paths}. Skipping batch.")
                        continue
                    
                    if loss is None or not torch.isfinite(loss) or processed_batch_size_in_step == 0:
                        if processed_batch_size_in_step == 0 and (loss is None or loss.item() == 0.0):
                            pass
                        else:
                            print(f"Non-finite val loss ({loss.item() if loss is not None else 'None'}) or zero processed samples. Skipping batch.")
                        continue

                    epoch_val_loss += loss.item() * processed_batch_size_in_step
                    processed_samples_in_val_epoch += processed_batch_size_in_step
                    correct_val_predictions += (torch.argmax(logits, dim=1) == labels[:processed_batch_size_in_step]).sum().item()
            
            avg_val_loss = epoch_val_loss / processed_samples_in_val_epoch if processed_samples_in_val_epoch > 0 else 0
            val_accuracy = correct_val_predictions / processed_samples_in_val_epoch if processed_samples_in_val_epoch > 0 else 0
            print(f"Epoch {epoch+1} Val: Avg Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f} (Processed: {processed_samples_in_val_epoch})")

            if val_accuracy > best_val_accuracy and processed_samples_in_val_epoch > 0:
                best_val_accuracy = val_accuracy
                best_model_path = os.path.join(args.output_dir, 't3_av_finetune_best_model.pt')
                torch.save({
                    'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'args': vars(args),
                    'val_accuracy': val_accuracy, 'train_accuracy': train_accuracy
                }, best_model_path)
                print(f"New best model saved to {best_model_path} with Val Accuracy: {val_accuracy:.4f}")

        # --- Checkpointing (Regularly) ---
        if (epoch + 1) % args.checkpoint_save_interval == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f't3_av_finetune_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy if val_loader else None, # Store last val accuracy
                'train_accuracy': train_accuracy, # Store last train accuracy
                'args': vars(args)
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Sound Classification Fine-tuning finished.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

if __name__ == '__main__':
    import numpy as np
    args = parse_args()
    main(args) 