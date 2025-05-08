import pandas as pd
import argparse
import os
import shutil

def copy_video_subset(
    subset_csv_path: str,
    source_video_dir: str,
    destination_video_dir: str,
    file_extension: str = ".mp4"
):
    """
    Reads a subset CSV, constructs video filenames, and copies them
    from a source directory to a destination directory.

    Args:
        subset_csv_path (str): Path to the CSV file listing the video subset.
        source_video_dir (str): Path to the directory containing the original full set of videos.
        destination_video_dir (str): Path to the directory where videos will be copied.
        file_extension (str): Extension of the video files (e.g., ".mp4").
    """
    try:
        # Read the CSV. Expecting columns: youtube_id,start_seconds,label,split
        df_subset = pd.read_csv(subset_csv_path, header=None, names=['youtube_id', 'start_seconds', 'label', 'split'])
        print(f"Successfully read {len(df_subset)} rows from {subset_csv_path}")
    except FileNotFoundError:
        print(f"Error: Subset CSV file not found at {subset_csv_path}")
        return
    except Exception as e:
        print(f"Error reading subset CSV file {subset_csv_path}: {e}")
        return

    if not os.path.isdir(source_video_dir):
        print(f"Error: Source video directory not found at {source_video_dir}")
        return

    # Create destination directory if it doesn't exist
    try:
        os.makedirs(destination_video_dir, exist_ok=True)
        print(f"Ensured destination directory exists: {destination_video_dir}")
    except OSError as e:
        print(f"Error creating destination directory {destination_video_dir}: {e}")
        return

    copied_count = 0
    skipped_count = 0

    for index, row in df_subset.iterrows():
        youtube_id = row['youtube_id']
        start_seconds = row['start_seconds']

        # Construct the source video filename
        # (e.g., ng44yvJYGCQ_000011.mp4)
        source_filename = f"{youtube_id}_{int(start_seconds):06d}{file_extension}"
        
        source_file_path = os.path.join(source_video_dir, source_filename)
        destination_file_path = os.path.join(destination_video_dir, source_filename)

        if os.path.exists(source_file_path):
            try:
                print(f"Copying '{source_file_path}' to '{destination_file_path}'...")
                shutil.copy2(source_file_path, destination_file_path) # copy2 preserves metadata
                copied_count += 1
            except Exception as e:
                print(f"Error copying file {source_filename}: {e}")
                skipped_count += 1
        else:
            print(f"Warning: Source file not found, skipping: {source_file_path}")
            skipped_count += 1
            
    print(f"\nCopy process finished.")
    print(f"Successfully copied {copied_count} files.")
    print(f"Skipped {skipped_count} files (not found or error during copy).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Copy a subset of video files based on a CSV list.")
    parser.add_argument(
        '--subset_csv',
        type=str,
        required=True,
        help='Path to the input CSV file listing the video subset (e.g., vggsound_subset.csv).'
    )
    parser.add_argument(
        '--source_video_dir',
        type=str,
        required=True,
        help='Path to the directory containing the original full set of video files.'
    )
    parser.add_argument(
        '--destination_video_dir',
        type=str,
        default="./video", # Default destination directory
        help='Path to the directory where videos will be copied. Defaults to ./video.'
    )
    parser.add_argument(
        '--file_extension',
        type=str,
        default=".mp4",
        help='Extension of the video files (e.g., ".mp4"). Default is ".mp4".'
    )
    args = parser.parse_args()

    print(f"Subset CSV: {args.subset_csv}")
    print(f"Source Video Directory: {args.source_video_dir}")
    print(f"Destination Video Directory: {args.destination_video_dir}")
    print(f"File Extension: {args.file_extension}")

    if not os.path.exists(args.subset_csv):
        print(f"Error: Subset CSV file '{args.subset_csv}' not found.")
    elif not os.path.isdir(args.source_video_dir):
         print(f"Error: Source video directory '{args.source_video_dir}' not found or is not a directory.")
    else:
        copy_video_subset(
            args.subset_csv,
            args.source_video_dir,
            args.destination_video_dir,
            args.file_extension
        ) 