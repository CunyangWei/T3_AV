# T3-AV Model Training and Fine-tuning Guide

This guide outlines the steps to pre-train and fine-tune the T3-AV model for audio-visual sound event classification.

## 目录 (Table of Contents)

- [T3-AV Model Training and Fine-tuning Guide](#t3-av-model-training-and-fine-tuning-guide)
  - [目录 (Table of Contents)](#目录-table-of-contents)
  - [1. 环境设置 (Environment Setup)](#1-环境设置-environment-setup)
  - [2. 数据准备 (Data Preparation)](#2-数据准备-data-preparation)
    - [2.1. 下载 VGGSound 数据集](#21-下载-vggsound-数据集)
    - [2.2. 创建数据子集 (Optional - Create Data Subset)](#22-创建数据子集-optional---create-data-subset)
  - [3. 模型训练阶段 (Model Training Stages)](#3-模型训练阶段-model-training-stages)
    - [3.1. Stage 1: 掩码自动编码器 (MAE) 预训练](#31-stage-1-掩码自动编码器-mae-预训练)
    - [3.2. Stage 2: 多模态对比学习](#32-stage-2-多模态对比学习)
    - [3.3. Stage 3: 音视频分类微调](#33-stage-3-音视频分类微调)
  - [4. 注意事项和进阶调整](#4-注意事项和进阶调整)

## 1. 环境设置 (Environment Setup)

确保您已经安装了所有必要的Python包。主要包括：

*   `torch`, `torchaudio`, `torchvision`
*   `transformers`
*   `pandas`
*   `opencv-python`
*   `moviepy`
*   `Pillow`
*   `numpy`
*   `scikit-learn` (for `accuracy_score`)
*   `tqdm`

此外还需要ffmpeg。
*   `apt-get install ffmpeg`

建议在conda或venv等虚拟环境中安装依赖。

## 2. 数据准备 (Data Preparation)

### 2.1. 下载 VGGSound 数据集

*   从官方渠道下载 VGGSound 数据集，包括视频文件和 `vggsound.csv` 注释文件。
*   解压视频文件到一个指定的目录，例如 `/data/vggsound/videos/`。
*   `vggsound.csv` 文件应包含视频ID、起始秒数、标签和数据集划分（train/test）等信息。

### 2.2. 创建数据子集 (Optional - Create Data Subset)

为了快速测试整个流程或在有限的计算资源下进行实验，您可以先创建一个较小的数据子集。使用我们提供的 `select_vggsound_subset.py` 脚本。

1.  **修改脚本中的默认输入路径**:
    打开 `select_vggsound_subset.py`，找到 `default=os.path.join(os.path.expanduser("~"), "vggsound.csv")` 这一行，将其修改为您实际的 `vggsound.csv` 文件路径。
2.  **运行脚本**:
    ```bash
    python select_vggsound_subset.py \
    --input_csv vggsound.csv \
    --output_csv vggsound_subset_10.csv \
    --num_classes 310 \
    --num_train 8 \
    --num_test 2 \
    --audio_dir /home/cunyang/scratch.bhatele-lab/vgg/full_dataset_extracted/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/ 
    ```
    在后续的训练命令中，您将使用这个新生成的子集CSV文件路径。

3.  **复制视频文件**:
    使用 `copy_subset_videos.py` 脚本复制视频文件到新的目录。
    ```bash
    python copy_subset_videos.py \
    --subset_csv vggsound_subset.csv \
    --source_video_dir /home/wcy/848m/VGGSound_dataset/full_dataset_extracted/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video \
    --destination_video_dir ./video
    ```
    保存子数据集。

python select_vggsound_classes.py \
    --input_csv vggsound.csv \
    --output_csv ./vggsound_10_classes_all_samples.csv \
    --num_classes 10 \
    --video_dir /home/cunyang/scratch.bhatele-lab/vgg/full_dataset_extracted/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/ \
    --seed 42

<!-- 3.  **清理数据**:
    使用 `clean_vggsound_csv.py` 脚本清理数据集，删除不存在的视频文件。
    ```bash
     python clean_vggsound_csv.py \
        --input_csv vggsound_subset.csv \
        --video_dir /home/wcy/848m/VGGSound_dataset/full_dataset_extracted/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video \
        --output_csv vggsound_cleaned.csv
    ```
    这将确保您的数据集只包含存在且可访问的视频文件。 -->
## 3. 模型训练阶段 (Model Training Stages)

模型训练分为三个主要阶段。您可以根据需求选择执行全部阶段或从某个阶段的预训练模型开始。

### 3.1. Stage 1: 掩码自动编码器 (MAE) 预训练

此阶段独立地对音频和视频模态进行掩码自动编码预训练，学习单模态的有效表示。

**命令示例**:
```bash
python train_stage1.py \
--csv_path ./vggsound_subset_40.csv \
--video_dir /home/cunyang/scratch.bhatele-lab/vgg/full_dataset_extracted/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/ \
--output_dir ./stage1_output \
--epochs 1 \
--batch_size 128 \
--learning_rate 1.5e-4 \
--warmup_epochs 5 \
--mask_ratio 0.75 \
--num_frames 16 \
--vit_model_name 'google/vit-base-patch16-224-in21k' \
--audio_duration 10.0 \
--bb_layers 6 \
--bb_heads 12 \
--dec_layers 2 \
--dec_heads 8 \
--device cuda \
--num_workers 16 \
--log_interval 50 \
--checkpoint_interval 5
```
*   **关键参数**:
    *   `--csv_path`: 训练数据CSV文件路径 (可以是完整数据集或子集)。
    *   `--video_dir`: 视频文件存放目录。
    *   `--output_dir`: Stage 1模型权重和日志的保存目录。
    *   `--train_video_mae` / `--train_audio_mae`: 开启对应模态的MAE训练。
*   训练完成后，最好的模型权重会保存在 `--output_dir` 中。

### 3.2. Stage 2: 多模态对比学习

此阶段使用Stage 1预训练的权重（可选，但推荐）进行初始化，通过对比学习对齐音频和视频表示。

**命令示例**:
```bash
python train_stage2.py \
--csv_path vggsound_subset.csv \
--video_dir ./video \
--stage1_checkpoint ./stage1_output/t3_av_stage1_epoch_5.pt \
--output_dir ./stage2_output \
--epochs 5 \
--batch_size 16 \
--learning_rate 1e-4 \
--warmup_epochs 3 \
--contrastive_dim 128 \
--temperature 0.07 \
--num_frames 16 \
--vit_model_name 'google/vit-base-patch16-224-in21k' \
--audio_duration 10.0 \
--bb_layers 6 \
--bb_heads 12 \
--proj_hidden_dim 768 \
--device cuda \
--num_workers 8 \
--log_interval 50 \
--checkpoint_interval 5
# --freeze_encoders \
# --freeze_backbone \

# python -m torch.distributed.run --nproc_per_node=1 train_stage2.py --ddp \
python train_stage2.py \
    --csv_path vggsound_subset_40.csv \
    --video_dir /home/cunyang/scratch.bhatele-lab/vgg/full_dataset_extracted/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/ \
    --stage1_checkpoint ./test_stage1_output/t3_av_stage1_epoch_50.pt \
    --output_dir ./stage2_output \
    --epochs 1 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --warmup_epochs 1 \
    --contrastive_dim 128 \
    --temperature 0.07 \
    --num_frames 16 \
    --vit_model_name 'google/vit-base-patch16-224-in21k' \
    --audio_duration 10.0 \
    --bb_layers 6 \
    --bb_heads 12 \
    --proj_hidden_dim 768 \
    --num_workers 16 \
    --log_interval 10 \
    --checkpoint_interval 1 \
    --amp \
    --compile_model \
    --compile_mode reduce-overhead
```
*   **关键参数**:
    *   `--stage1_checkpoint`: Stage 1训练好的模型权重路径。如果跳过Stage 1，这里可以是一个基础的ViT权重，但模型结构需要对应。
    *   `--output_dir`: Stage 2模型权重和日志的保存目录。
    *   `--freeze_encoders` / `--freeze_backbone`: 可选，用于冻结部分网络层。
*   训练完成后，最好的模型权重会保存在 `--output_dir` 中。


### 3.3. Stage 3: 音视频分类微调

此阶段使用Stage 2（或Stage 1）预训练的权重初始化模型，并在目标下游任务（VGGSound声音事件分类）上进行微调。这里使用音频和视频特征融合进行分类。

**命令示例**:
```bash
python train_finetune.py \
--csv_path vggsound_subset.csv \
--video_dir ./video \
--pretrained_checkpoint ./stage2_output/t3_av_stage2_epoch_5.pt \
--output_dir ./finetune_av_classification_output \
--epochs 5 \
--batch_size 16 \
--learning_rate 1e-4 \
--warmup_epochs 3 \
--unfreeze_backbone_layers 2 \
--unfreeze_encoder_layers 1 \
--num_frames 16 \
--vit_model_name 'google/vit-base-patch16-224-in21k' \
--audio_duration 10.0 \
--bb_layers 6 \
--bb_heads 12 \
--clf_hidden_dim 512 \
--device cuda \
--num_workers 8 \
--log_interval 20 \
--eval_interval 1 \
--checkpoint_save_interval 1
# 或者完全只微调分类头: --freeze_all_except_head \
```
*   **关键参数**:
    *   `--pretrained_checkpoint`: Stage 2（或Stage 1）训练好的模型权重路径。
    *   `--output_dir`: 微调后模型权重和日志的保存目录。
    *   `--freeze_all_except_head`: 只训练最后的分类头。
    *   `--unfreeze_backbone_layers`: 解冻共享骨干网络的最后几层。
    *   `--unfreeze_encoder_layers`: 解冻音频编码器ViT的最后几层（视频编码器通常保持冻结，因为主要依赖音频特征，视频起辅助作用）。
    *   学习率 (`--learning_rate`) 通常比预训练阶段要小。
*   训练过程中会进行验证，并保存验证集上表现最好的模型 (`t3_av_finetune_best_model.pt`) 和每个checkpoint间隔的模型。

## 4. 注意事项和进阶调整

*   **路径配置**: 务必将脚本中所有 `/path/to/your/...` 替换为您的实际文件和目录路径。
*   **模型配置参数**: 在运行`train_stage2.py`和`train_finetune.py`时，提供的模型结构参数（如`--bb_layers`, `--num_frames`等）必须与加载的预训练检查点 (`.pt`文件) 所使用的模型结构一致，否则权重加载会失败或出现非预期行为。检查点本身也保存了训练时的参数，脚本会尝试使用它们，但命令行参数具有更高优先级。
*   **显存 (GPU Memory)**: 根据您的GPU显存调整 `--batch_size`。如果遇到显存不足 (CUDA out of memory) 的错误，请减小批次大小。
*   **训练时间**: 在完整VGGSound数据集上训练可能非常耗时。使用数据子集进行初步实验是个好主意。
*   **超参数调优**: 上述命令中的学习率、epoch数、冻结策略等均为示例值，您可能需要根据实际情况进行调整以获得最佳性能。
*   **错误处理**: 训练脚本包含基本的错误处理，例如跳过无法处理的视频批次。请留意控制台输出的警告和错误信息。
*   **数据验证**: 在开始训练前，强烈建议先使用少量数据（例如，`vggsound_dataset.py`中的 `__main__` 测试部分）来验证数据加载和预处理流程是否正常工作。

