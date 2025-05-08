python train_stage1.py \
    --csv_path ./vggsound.csv \
    --video_dir /home/wcy/848m/VGGSound_dataset/full_dataset_extracted/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video \
    --output_dir ./stage1_output \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 1.5e-4 \
    --warmup_epochs 10 \
    --num_workers 4 \
    --device cuda \
    --checkpoint_interval 5 \
    # --- Add/adjust other args as needed ---
    # e.g., --num_frames 16 --audio_duration 10.0 
    # Use --no_train_video_mae or --no_train_audio_mae to disable one loss