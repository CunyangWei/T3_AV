#!/bin/bash

#SBATCH -p gpu 
#SBATCH -A cmsc828-class
#SBATCH -J spmm
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --gpus=h100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g

module load ffmpeg
source /scratch/zt1/project/bhatele-lab/user/cunyang/t3_venv/bin/activate

NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4))
echo $NNODES
echo $GPUS

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_CROSS_NIC=1
# export NCCL_SOCKET_IFNAME=hsn
export MPICH_GPU_SUPPORT_ENABLED=0

# CSV_PATH="YOUR_PATH_TO_VGGSOUND_SUBSET.CSV"
# VIDEO_DIR="YOUR_PATH_TO_VIDEO_DIR"
# PRETRAINED_CKPT="YOUR_PATH_TO_PRETRAINED_CHECKPOINT.pt"
# OUTPUT_DIR="YOUR_OUTPUT_DIR_FINETUNE"

# BATCH_SIZE=32
# LEARNING_RATE=5e-5
# EPOCHS=30
# NUM_WORKERS=8

# echo "Running Fine-tuning with DDP"

# python -m torch.distributed.run --nproc_per_node=4 train_finetune.py --ddp \
#     --csv_path ${CSV_PATH} \
#     --video_dir ${VIDEO_DIR} \
#     --pretrained_checkpoint ${PRETRAINED_CKPT} \
#     --output_dir ${OUTPUT_DIR} \
#     --epochs ${EPOCHS} \
#     --batch_size ${BATCH_SIZE} \
#     --learning_rate ${LEARNING_RATE} \
#     --num_workers ${NUM_WORKERS} \
#     --amp \
#     --log_interval 50 \
#     --eval_interval 1 \
#     --checkpoint_save_interval 1 \

# echo "Fine-tuning script finished."


python -m torch.distributed.run --nproc_per_node=4 train_finetune.py --ddp \
    --csv_path ./vggsound_10_classes_all_samples.csv \
    --video_dir /home/cunyang/scratch.bhatele-lab/vgg/full_dataset_extracted/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/ \
    --pretrained_checkpoint /home/cunyang/scratch.bhatele-lab/T3_pt/allgather40_stage2_output/t3_av_stage2_epoch_50.pt \
    --output_dir ./allgather40_10_finetune_output_freeze_encoder_lowrate \
    --epochs 30 \
    --batch_size 20 \
    --learning_rate 1e-5 \
    --warmup_epochs 3 \
    --num_frames 16 \
    --vit_model_name 'google/vit-base-patch16-224-in21k' \
    --audio_duration 10.0 \
    --bb_layers 6 \
    --bb_heads 12 \
    --clf_hidden_dim 512 \
    --dec_layers 2 \
    --dec_heads 8 \
    --proj_hidden_dim_contrastive 768 \
    --contrastive_dim 128 \
    --num_workers 16 \
    --log_interval 5 \
    --checkpoint_save_interval 5 \
    --eval_interval 1 \
    --freeze_encoders \
    --compile_model \
    --compile_mode 'reduce-overhead'