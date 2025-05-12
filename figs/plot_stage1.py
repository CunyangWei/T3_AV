import matplotlib.pyplot as plt
import re

def parse_log_data(log_file_path):
    """
    解析日志文件，提取 Epoch 和损失值。

    Args:
        log_file_path (str): 日志文件的路径。

    Returns:
        tuple: 包含四个列表的元组 (epochs, avg_losses, video_mae_losses, audio_mae_losses)
    """
    epochs = []
    avg_losses = []
    video_mae_losses = []
    audio_mae_losses = []

    # 正则表达式匹配日志行
    epoch_pattern = re.compile(r"Epoch (\d+) Finished.")
    avg_loss_pattern = re.compile(r"Average Loss \(Rank 0\): (\d+\.\d+)")
    video_mae_loss_pattern = re.compile(r"Average Video MAE Loss \(Rank 0\): (\d+\.\d+)")
    audio_mae_loss_pattern = re.compile(r"Average Audio MAE Loss \(Rank 0\): (\d+\.\d+)")

    with open(log_file_path, 'r') as f:
        log_content = f.read()

    # 按 "Epoch ... Finished." 分割每个 Epoch 的数据块
    epoch_blocks = epoch_pattern.split(log_content)[1:] # 第一个元素是空字符串，所以跳过

    for i in range(0, len(epoch_blocks), 2):
        epoch_num_str = epoch_blocks[i]
        block_data = epoch_blocks[i+1]

        epochs.append(int(epoch_num_str))

        avg_loss_match = avg_loss_pattern.search(block_data)
        if avg_loss_match:
            avg_losses.append(float(avg_loss_match.group(1)))
        else:
            avg_losses.append(None) # 如果找不到，则添加 None

        video_mae_loss_match = video_mae_loss_pattern.search(block_data)
        if video_mae_loss_match:
            video_mae_losses.append(float(video_mae_loss_match.group(1)))
        else:
            video_mae_losses.append(None)

        audio_mae_loss_match = audio_mae_loss_pattern.search(block_data)
        if audio_mae_loss_match:
            audio_mae_losses.append(float(audio_mae_loss_match.group(1)))
        else:
            audio_mae_losses.append(None)

    return epochs, avg_losses, video_mae_losses, audio_mae_losses

def plot_losses(epochs, avg_losses, video_mae_losses, audio_mae_losses):
    """
    绘制损失曲线图。

    Args:
        epochs (list): Epoch 编号列表。
        avg_losses (list): 平均总损失列表。
        video_mae_losses (list): 平均视频 MAE 损失列表。
        audio_mae_losses (list): 平均音频 MAE 损失列表。
    """
    plt.figure(figsize=(12, 6))

    plt.plot(epochs, avg_losses, label='Average Loss', marker='o')
    plt.plot(epochs, video_mae_losses, label='Average Video MAE Loss', marker='s')
    plt.plot(epochs, audio_mae_losses, label='Average Audio MAE Loss', marker='^')

    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs) # 确保每个 epoch 都有刻度
    plt.tight_layout() # 调整布局以防止标签重叠
    plt.savefig('stage1_40.png')

# --- 主程序 ---
if __name__ == "__main__":
    log_file = "stage1_40.txt"  # 替换为您的日志文件名

    # --- 为演示创建一个示例 log.txt 文件 ---
    # 如果您已经有 log.txt 文件，请删除或注释掉下面的代码块
    sample_log_data = ""
    for i in range(15): # 生成50个 Epoch 的示例数据
        epoch_num = i
        avg_loss = 0.5 - i * 0.005 + (0.01 * (-1)**i) # 示例数据，模拟下降趋势和一些波动
        video_loss = 0.4 - i * 0.004 + (0.008 * (-1)**i)
        audio_loss = 0.1 - i * 0.001 + (0.002 * (-1)**i)
        sample_log_data += f"Epoch {epoch_num} Finished.\n"
        sample_log_data += f"  Average Loss (Rank 0): {avg_loss:.4f}\n"
        sample_log_data += f"  Average Video MAE Loss (Rank 0): {video_loss:.4f}\n"
        sample_log_data += f"  Average Audio MAE Loss (Rank 0): {audio_loss:.4f}\n\n"

    # 检查是否需要创建示例文件
    try:
        with open(log_file, 'r') as f:
            pass # 文件存在，不做任何事
    except FileNotFoundError:
        print(f"'{log_file}' not found. Creating a sample log file for demonstration.")
        with open(log_file, 'w') as f:
            f.write(sample_log_data)
        print(f"Sample '{log_file}' created. Please replace it with your actual log file or modify the script.")
    # --- 示例文件创建结束 ---

    epochs, avg_losses, video_mae_losses, audio_mae_losses = parse_log_data(log_file)

    if not epochs:
        print("No data parsed. Please check your log file format and path.")
    else:
        print(f"Parsed {len(epochs)} epochs.")
        plot_losses(epochs, avg_losses, video_mae_losses, audio_mae_losses)