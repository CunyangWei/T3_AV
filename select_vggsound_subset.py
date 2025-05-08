import pandas as pd
import argparse
import os
import random

def select_subset(
    csv_file_path: str,
    output_csv_path: str,
    num_classes_to_select: int,
    num_train_per_class: int,
    num_test_per_class: int,
    video_dir_path: str,
    file_extension: str = ".mp4"
):
    """
    Reads the VGGSound CSV, randomly selects a specified number of classes,
    then for each class, selects a specified number of train and test samples
    if their corresponding video files exist.
    Saves the subset to a new CSV file.

    Args:
        csv_file_path (str): Path to the full vggsound.csv file.
        output_csv_path (str): Path where the new subset CSV will be saved.
        num_classes_to_select (int): The number of unique classes to select.
        num_train_per_class (int): Number of train samples to select per class.
        num_test_per_class (int): Number of test samples to select per class.
        video_dir_path (str): Directory containing the video files.
        file_extension (str): Extension of the video files (e.g., ".wav").
    """
    try:
        df = pd.read_csv(csv_file_path, header=None, names=['youtube_id', 'start_seconds', 'label', 'split'])
        print(f"Successfully read {len(df)} rows from {csv_file_path}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file {csv_file_path}: {e}")
        return

    if 'label' not in df.columns:
        print(f"Error: 'label' column not found. Columns: {df.columns.tolist()}")
        return
    if not os.path.isdir(video_dir_path):
        print(f"Error: video directory not found at {video_dir_path}")
        return

    unique_labels = df['label'].unique()
    print(f"Found {len(unique_labels)} unique labels in the dataset.")

    if len(unique_labels) < num_classes_to_select:
        print(f"Warning: Requested {num_classes_to_select} classes, but only {len(unique_labels)} available. Selecting all.")
        selected_classes = unique_labels.tolist()
    else:
        selected_classes = random.sample(list(unique_labels), num_classes_to_select)
    print(f"Randomly selected {len(selected_classes)} classes: {selected_classes}")

    final_selected_rows = []

    for s_class in selected_classes:
        class_df = df[df['label'] == s_class]

        # Process train samples
        train_samples_for_class = class_df[class_df['split'] == 'train']
        eligible_train_samples_rows = []
        for index, row in train_samples_for_class.iterrows():
            # Construct filename as ID_startseconds_padded.extension
            filename = f"{row['youtube_id']}_{int(row['start_seconds']):06d}{file_extension}"
            potential_file_path = os.path.join(video_dir_path, filename)
            print(f"Checking for TRAIN file: {potential_file_path}")
            if os.path.exists(potential_file_path):
                eligible_train_samples_rows.append(row)
        
        eligible_train_df = pd.DataFrame(eligible_train_samples_rows)
        
        if not eligible_train_df.empty:
            if len(eligible_train_df) < num_train_per_class:
                print(f"Warning: For class '{s_class}' (train), only {len(eligible_train_df)} existing samples found, requested {num_train_per_class}. Taking all available.")
                final_selected_rows.extend(eligible_train_df.to_dict('records'))
            else:
                # Use random_state for reproducibility if desired, e.g., random_state=42
                selected_train = eligible_train_df.sample(n=num_train_per_class) 
                final_selected_rows.extend(selected_train.to_dict('records'))

        # Process test samples
        test_samples_for_class = class_df[class_df['split'] == 'test']
        eligible_test_samples_rows = []
        for index, row in test_samples_for_class.iterrows():
            # Construct filename as ID_startseconds_padded.extension
            filename = f"{row['youtube_id']}_{int(row['start_seconds']):06d}{file_extension}"
            potential_file_path = os.path.join(video_dir_path, filename)
            print(f"Checking for TEST file: {potential_file_path}")
            if os.path.exists(potential_file_path):
                eligible_test_samples_rows.append(row)

        eligible_test_df = pd.DataFrame(eligible_test_samples_rows)

        if not eligible_test_df.empty:
            if len(eligible_test_df) < num_test_per_class:
                print(f"Warning: For class '{s_class}' (test), only {len(eligible_test_df)} existing samples found, requested {num_test_per_class}. Taking all available.")
                final_selected_rows.extend(eligible_test_df.to_dict('records'))
            else:
                selected_test = eligible_test_df.sample(n=num_test_per_class)
                final_selected_rows.extend(selected_test.to_dict('records'))
                
    if not final_selected_rows:
        print("Warning: The final subset is empty. No samples met the criteria.")
        # Create an empty DataFrame with correct columns to avoid error on to_csv
        df_subset = pd.DataFrame(columns=['youtube_id', 'start_seconds', 'label', 'split'])
    else:
        df_subset = pd.DataFrame(final_selected_rows)
        # Ensure correct column order if creating from list of dicts
        if not df_subset.empty:
             df_subset = df_subset[['youtube_id', 'start_seconds', 'label', 'split']]
        print(f"Created a subset with {len(df_subset)} samples.")

    try:
        df_subset.to_csv(output_csv_path, index=False, header=False)
        print(f"Subset CSV saved to {output_csv_path}")
    except Exception as e:
        print(f"Error saving subset CSV to {output_csv_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select a subset of classes and samples from VGGSound CSV, checking for file existence.")
    parser.add_argument(
        '--input_csv',
        type=str,
        default="vggsound.csv",
        help='Path to the input vggsound.csv file.'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default="vggsound_subset_selected.csv",
        help='Path to save the output subset CSV file.'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=10,
        help='Number of unique classes to randomly select.'
    )
    parser.add_argument(
        '--num_train',
        type=int,
        default=5,
        help='Number of train samples to select per chosen class (if files exist).'
    )
    parser.add_argument(
        '--num_test',
        type=int,
        default=2,
        help='Number of test samples to select per chosen class (if files exist).'
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        required=True,
        help='Path to the directory containing video files (e.g., wav, mp3).'
    )
    parser.add_argument(
        '--file_extension',
        type=str,
        default=".mp4",
        help='Extension of the video files (e.g., ".wav", ".mp3"). Default is ".mp4".'
    )
    args = parser.parse_args()

    print(f"Input CSV: {args.input_csv}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Number of classes to select: {args.num_classes}")
    print(f"Number of train samples per class: {args.num_train}")
    print(f"Number of test samples per class: {args.num_test}")
    print(f"video directory: {args.video_dir}")
    print(f"video file extension: {args.file_extension}")

    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV file '{args.input_csv}' not found.")
    elif not os.path.isdir(args.video_dir):
        print(f"Error: video directory '{args.video_dir}' not found or is not a directory.")
    else:
        select_subset(
            args.input_csv,
            args.output_csv,
            args.num_classes,
            args.num_train,
            args.num_test,
            args.video_dir,
            args.file_extension
        ) 