    python train_stage2.py \
        --csv_path ./vggsound.csv \
        --video_dir /home/wcy/848m/VGGSound_dataset/full_dataset_extracted/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video \
        --stage1_checkpoint ./stage1_output/t3_av_stage1_epoch_100.pt \
        --output_dir ./stage2_output \
        --epochs 50 \
        --batch_size 32 \
        --learning_rate 1e-5 \
        --warmup_epochs 5 \
        --num_workers 4 \
        --device cuda \
        --contrastive_dim 128 \
        --temperature 0.07 \
        # --- Optional Freezing ---
        # --freeze_encoders \
        # --freeze_backbone \
        # --- Model config args (ensure they match stage 1 checkpoint or model init) ---
        # --num_frames 16 \
        # --audio_duration 10.0 \ 
        # --bb_layers 6 \
        # --bb_heads 12 \
        # --proj_hidden_dim 768 \
        # --vit_model_name 'google/vit-base-patch16-224-in21k' 