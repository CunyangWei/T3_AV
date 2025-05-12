import pandas as pd
import argparse
import os
import random

def select_subset(csv_file_path: str, output_csv_path: str, num_classes_to_select: int,
                  num_train_per_class: int, num_test_per_class: int, audio_dir: str):
    """
    Reads the VGGSound CSV, randomly selects a specified number of classes,
    filters samples for these classes based on 'train'/'test' splits and video file existence,
    and saves the subset to a new CSV file.

    Args:
        csv_file_path (str): Path to the full vggsound.csv file.
        output_csv_path (str): Path where the new subset CSV will be saved.
        num_classes_to_select (int): The number of unique classes to select.
        num_train_per_class (int): Number of training samples to select per class.
        num_test_per_class (int): Number of testing samples to select per class.
        audio_dir (str): Path to the directory containing video files.
    """
    try:
        # Read the CSV. Assuming format: youtube_id,start_seconds,label,split
        # Adjust names if your CSV has a different header or no header.
        df = pd.read_csv(csv_file_path, header=None, names=['youtube_id', 'start_seconds', 'label', 'split'])
        print(f"Successfully read {len(df)} rows from {csv_file_path}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        print("Please provide the correct path to your vggsound.csv file.")
        return
    except Exception as e:
        print(f"Error reading CSV file {csv_file_path}: {e}")
        return

    # Ensure the 'label' and 'split' columns exist
    if 'label' not in df.columns:
        print(f"Error: 'label' column not found in the CSV. Columns found: {df.columns.tolist()}")
        print("Please ensure your CSV is formatted correctly with a 'label' column (or adjust column names in the script).")
        return
    if 'split' not in df.columns:
        print(f"Error: 'split' column not found in the CSV. Columns found: {df.columns.tolist()}")
        print("Please ensure your CSV is formatted correctly with a 'split' column.")
        return

    if not os.path.isdir(audio_dir):
        print(f"Error: Audio directory '{audio_dir}' not found or is not a directory.")
        return

    # Get unique labels
    unique_labels = df['label'].unique()
    print(f"Found {len(unique_labels)} unique labels in the dataset.")

    if len(unique_labels) < num_classes_to_select:
        print(f"Warning: Requested {num_classes_to_select} classes, but only {len(unique_labels)} unique classes are available. Selecting all available classes.")
        selected_class_names = unique_labels.tolist()
    else:
        # Use a fixed random seed for class selection if reproducibility is desired here too
        # random.seed(42) 
        selected_class_names = random.sample(list(unique_labels), num_classes_to_select)

    print(f"Randomly selected {len(selected_class_names)} classes: {selected_class_names}")

    all_selected_samples_list = []

    for class_label in selected_class_names:
        print(f"\nProcessing class: {class_label}")
        class_df = df[df['label'] == class_label]

        # Training samples
        train_df_class_split = class_df[class_df['split'] == 'train']
        print(f"  Found {len(train_df_class_split)} potential training samples for class '{class_label}' (split='train').")
        
        actual_train_samples_rows = []
        for _, row in train_df_class_split.iterrows():
            video_filename = f"{row['youtube_id']}_{int(row['start_seconds']):06d}.mp4"
            video_path = os.path.join(audio_dir, video_filename)
            if os.path.exists(video_path):
                actual_train_samples_rows.append(row)
        
        df_available_train = pd.DataFrame(actual_train_samples_rows)
        if not df_available_train.empty:
             # Ensure columns are consistent with the original df
            df_available_train = df_available_train[df.columns]


        print(f"  Found {len(df_available_train)} training samples with existing videos for class '{class_label}'.")

        if not df_available_train.empty and num_train_per_class > 0:
            if len(df_available_train) > num_train_per_class:
                selected_train_for_class = df_available_train.sample(n=num_train_per_class, random_state=42)
            else:
                selected_train_for_class = df_available_train
            print(f"  Selected {len(selected_train_for_class)} training samples for class '{class_label}'.")
            if not selected_train_for_class.empty:
                all_selected_samples_list.append(selected_train_for_class)
        elif num_train_per_class > 0:
             print(f"  Selected 0 training samples for class '{class_label}' (none available or found).")


        # Testing samples
        test_df_class_split = class_df[class_df['split'] == 'test']
        print(f"  Found {len(test_df_class_split)} potential testing samples for class '{class_label}' (split='test').")

        actual_test_samples_rows = []
        for _, row in test_df_class_split.iterrows():
            video_filename = f"{row['youtube_id']}_{int(row['start_seconds']):06d}.mp4"
            video_path = os.path.join(audio_dir, video_filename)
            if os.path.exists(video_path):
                actual_test_samples_rows.append(row)

        df_available_test = pd.DataFrame(actual_test_samples_rows)
        if not df_available_test.empty:
            df_available_test = df_available_test[df.columns]


        print(f"  Found {len(df_available_test)} testing samples with existing videos for class '{class_label}'.")
        
        if not df_available_test.empty and num_test_per_class > 0:
            if len(df_available_test) > num_test_per_class:
                selected_test_for_class = df_available_test.sample(n=num_test_per_class, random_state=42)
            else:
                selected_test_for_class = df_available_test
            print(f"  Selected {len(selected_test_for_class)} testing samples for class '{class_label}'.")
            if not selected_test_for_class.empty:
                all_selected_samples_list.append(selected_test_for_class)
        elif num_test_per_class > 0:
            print(f"  Selected 0 testing samples for class '{class_label}' (none available or found).")


    if not all_selected_samples_list:
        print("\nWarning: The final subset is empty. No samples met all criteria or requested counts were zero.")
        df_subset = pd.DataFrame(columns=['youtube_id', 'start_seconds', 'label', 'split'])
    else:
        df_subset = pd.concat(all_selected_samples_list).reset_index(drop=True)
        # Ensure original column order again after concat
        df_subset = df_subset[['youtube_id', 'start_seconds', 'label', 'split']]
        print(f"\nCreated a final subset with {len(df_subset)} samples in total.")

    # Save the subset to a new CSV file without header and index
    try:
        df_subset.to_csv(output_csv_path, index=False, header=False)
        print(f"Subset CSV saved to {output_csv_path}")
    except Exception as e:
        print(f"Error saving subset CSV to {output_csv_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select a subset of classes and samples from VGGSound CSV, checking for video file existence.")
    parser.add_argument(
        '--input_csv',
        type=str,
        default=os.path.join(os.path.expanduser("~"), "vggsound.csv"),
        help='Path to the input vggsound.csv file.'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default="vggsound_subset_custom.csv", # Changed default name slightly
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
        help='Number of training samples to select per class (after video check). Set to 0 to skip.'
    )
    parser.add_argument(
        '--num_test',
        type=int,
        default=2,
        help='Number of testing samples to select per class (after video check). Set to 0 to skip.'
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        required=True,
        help='Path to the directory containing video files. Expected format in dir: {youtube_id}_{start_seconds:06d}.mp4.'
    )
    args = parser.parse_args()

    print(f"Input CSV: {args.input_csv}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Number of classes to select: {args.num_classes}")
    print(f"Number of train samples per class: {args.num_train}")
    print(f"Number of test samples per class: {args.num_test}")
    print(f"Audio directory for video check: {args.audio_dir}")

    # Check if input CSV exists before proceeding
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV file '{args.input_csv}' not found.")
        print("Please ensure the --input_csv argument points to your actual vggsound.csv file,")
        print("or modify the 'default' value in the script if running without CLI arguments.")
    else:
        select_subset(args.input_csv, args.output_csv, args.num_classes,
                      args.num_train, args.num_test, args.audio_dir) 