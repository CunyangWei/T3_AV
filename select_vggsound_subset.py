import pandas as pd
import argparse
import os
import random

def select_subset(csv_file_path: str, output_csv_path: str, num_classes_to_select: int):
    """
    Reads the VGGSound CSV, randomly selects a specified number of classes,
    filters the DataFrame to include only samples from these classes,
    and saves the subset to a new CSV file.

    Args:
        csv_file_path (str): Path to the full vggsound.csv file.
        output_csv_path (str): Path where the new subset CSV will be saved.
        num_classes_to_select (int): The number of unique classes to select.
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

    # Ensure the 'label' column exists
    if 'label' not in df.columns:
        print(f"Error: 'label' column not found in the CSV. Columns found: {df.columns.tolist()}")
        print("Please ensure your CSV is formatted correctly with a 'label' column (or adjust column names in the script).")
        return

    # Get unique labels
    unique_labels = df['label'].unique()
    print(f"Found {len(unique_labels)} unique labels in the dataset.")

    if len(unique_labels) < num_classes_to_select:
        print(f"Warning: Requested {num_classes_to_select} classes, but only {len(unique_labels)} unique classes are available. Selecting all available classes.")
        selected_classes = unique_labels.tolist()
    else:
        selected_classes = random.sample(list(unique_labels), num_classes_to_select)

    print(f"Randomly selected {len(selected_classes)} classes: {selected_classes}")

    # Filter the DataFrame to include only rows with the selected labels
    df_subset = df[df['label'].isin(selected_classes)]

    if df_subset.empty:
        print("Warning: The subset is empty. This might happen if selected classes have no samples or an issue occurred.")
    else:
        print(f"Created a subset with {len(df_subset)} samples belonging to the selected classes.")

    # Save the subset to a new CSV file without header and index
    try:
        df_subset.to_csv(output_csv_path, index=False, header=False)
        print(f"Subset CSV saved to {output_csv_path}")
    except Exception as e:
        print(f"Error saving subset CSV to {output_csv_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select a subset of classes from VGGSound CSV.")
    parser.add_argument(
        '--input_csv',
        type=str,
        # !!! IMPORTANT: Replace with the ACTUAL path to your vggsound.csv if not providing via CLI !!!
        default=os.path.join(os.path.expanduser("~"), "vggsound.csv"),
        help='Path to the input vggsound.csv file.'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default="vggsound_subset_10_classes.csv",
        help='Path to save the output subset CSV file.'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=10,
        help='Number of unique classes to randomly select.'
    )
    args = parser.parse_args()

    print(f"Input CSV: {args.input_csv}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Number of classes to select: {args.num_classes}")

    # Check if input CSV exists before proceeding
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV file '{args.input_csv}' not found.")
        print("Please ensure the --input_csv argument points to your actual vggsound.csv file,")
        print("or modify the 'default' value in the script if running without CLI arguments.")
    else:
        select_subset(args.input_csv, args.output_csv, args.num_classes) 