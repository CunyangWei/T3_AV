#!/bin/bash

#SBATCH -p gpu 
#SBATCH -A bhatele-lab-cmsc  
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


# srun ./get_rank.sh python train_stage1.py --ddp \
python -m torch.distributed.run --nproc_per_node=4 train_stage1.py --ddp \
--csv_path ./vggsound_subset_40.csv \
--video_dir /home/cunyang/scratch.bhatele-lab/vgg/full_dataset_extracted/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/ \
--output_dir ./test85_40_stage1_output \
--epochs 20 \
--batch_size 96 \
--learning_rate 1.5e-4 \
--warmup_epochs 2 \
--mask_ratio 0.85 \
--num_frames 16 \
--vit_model_name 'google/vit-base-patch16-224-in21k' \
--audio_duration 10.0 \
--bb_layers 6 \
--bb_heads 12 \
--dec_layers 2 \
--dec_heads 8 \
--device cuda \
--num_workers 16 \
--log_interval 5 \
--checkpoint_interval 5