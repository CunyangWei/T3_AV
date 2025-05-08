import pandas as pd
import argparse
import os
from tqdm import tqdm # For progress bar

def clean_csv(input_csv_path: str, video_dir_path: str, output_csv_path: str):
    """
    Reads a VGGSound-style CSV, checks if the corresponding video file exists
    for each row based on the video directory, and writes only the rows
    with existing video files to a new output CSV.

    Args:
        input_csv_path (str): Path to the input CSV file (e.g., vggsound.csv or a subset).
                              Expected format (no header): youtube_id,start_seconds,label,split
        video_dir_path (str): Path to the directory containing the actual video files.
        output_csv_path (str): Path where the cleaned CSV file will be saved.
    """
    expected_columns = ['youtube_id', 'start_seconds', 'label', 'split']
    try:
        df = pd.read_csv(input_csv_path, header=None, names=expected_columns)
        print(f"Successfully read {len(df)} rows from {input_csv_path}")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file {input_csv_path}: {e}")
        return

    if not os.path.isdir(video_dir_path):
         print(f"Error: Video directory not found at {video_dir_path}")
         print("Please provide the correct path to the directory containing video files.")
         return

    valid_rows_indices = []
    print(f"Checking for video file existence in directory: {video_dir_path}...")

    # Iterate through rows and check if the corresponding video file exists
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Checking files"):
        try:
            yt_id = row['youtube_id']
            # Ensure start_seconds is treated as integer for formatting
            start_sec = int(row['start_seconds'])

            # Construct the expected filename format
            video_filename = f"{yt_id}_{start_sec:06d}.mp4"
            full_video_path = os.path.join(video_dir_path, video_filename)

            if os.path.exists(full_video_path):
                valid_rows_indices.append(index)
            # Optional: uncomment below to print warnings for missing files
            # else:
            #     print(f"Warning: Video file not found for row {index}: {full_video_path}")

        except (TypeError, ValueError) as e:
            print(f"Warning: Skipping row {index} due to invalid data (youtube_id='{row.get('youtube_id', 'N/A')}', start_seconds='{row.get('start_seconds', 'N/A')}'): {e}")
        except Exception as e:
            print(f"Warning: An unexpected error occurred processing row {index}: {e}")


    # Create a new DataFrame with only the valid rows
    df_cleaned = df.loc[valid_rows_indices]

    num_original = len(df)
    num_cleaned = len(df_cleaned)
    num_removed = num_original - num_cleaned

    print(f"\nCheck complete.")
    print(f"  Original rows: {num_original}")
    print(f"  Valid rows (video found): {num_cleaned}")
    print(f"  Rows removed (video not found or error): {num_removed}")

    if num_cleaned == 0:
        print("Warning: No valid video files found for the entries in the CSV. The output CSV will be empty.")
        # Still save an empty file for consistency, or you could choose to exit here.

    # Save the cleaned DataFrame to the output CSV
    try:
        df_cleaned.to_csv(output_csv_path, index=False, header=False)
        print(f"Cleaned CSV saved successfully to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving cleaned CSV to {output_csv_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean a VGGSound CSV file by removing rows with non-existent video files.")
    parser.add_argument(
        '--input_csv',
        type=str,
        required=True,
        help='Path to the input vggsound.csv file (or subset).'
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        required=True,
        help='Path to the directory containing the actual video files (.mp4).'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        required=True,
        help='Path to save the output cleaned CSV file.'
    )
    args = parser.parse_args()

    clean_csv(args.input_csv, args.video_dir, args.output_csv) 