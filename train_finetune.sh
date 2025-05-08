python train_finetune.py \
    --csv_path /path/to/your/vggsound.csv \
    --video_dir /path/to/your/videos \
    --pretrained_checkpoint /path/to/your/stage2_output/t3_av_stage2_epoch_XX.pt \
    --output_dir ./finetune_sound_classification_output \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --warmup_epochs 3 \
    --freeze_all_except_head \
    # Or, for more granular unfreezing:
    # --unfreeze_backbone_layers 2 \
    # --unfreeze_encoder_layers 1 \
    --clf_hidden_dim 512 \
    --device cuda \
    --num_workers 8 \
    --bb_layers 6 \
    --bb_heads 12 \
    # ... other relevant model config args used during pretraining ...