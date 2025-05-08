import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class VGGSoundDataset(Dataset):
    """VGGSound dataset for loading video paths and labels."""

    def __init__(self, csv_file_path: str, video_dir: str, split: str = 'train', 
                 label_to_idx: dict = None, transform=None):
        """
        Args:
            csv_file_path (str): Path to the vggsound.csv file.
            video_dir (str): Directory containing the video files (.mp4).
            split (str): 'train', 'test', or 'all' to select dataset split.
            label_to_idx (dict, optional): A mapping from string labels to integer indices.
                                         If None, it will be created from the unique labels in the split.
            transform (callable, optional): Optional transform to be applied on a sample.
                                           (Likely not used if model takes file paths directly)
        """
        self.video_dir = video_dir
        self.transform = transform
        self.split = split

        try:
            df = pd.read_csv(csv_file_path, header=None, names=['youtube_id', 'start_seconds', 'label', 'split'])
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_file_path}")
            print("Please provide the correct path to vggsound.csv")
            self.samples = []
            self.label_to_idx = {}
            self.idx_to_label = {}
            self.num_classes = 0
            return
        except Exception as e:
            print(f"Error reading CSV file {csv_file_path}: {e}")
            self.samples = []
            self.label_to_idx = {}
            self.idx_to_label = {}
            self.num_classes = 0
            return

        # Clean up column names if they have extra spaces from "# YouTube ID"
        df.columns = df.columns.str.strip()
        if '# YouTube ID' in df.columns:
            df.rename(columns={'# YouTube ID': 'youtube_id'}, inplace=True)
        
        # Filter by split (train or test)
        # The CSV has 'train' or 'test' in the split column.
        if self.split != 'all':
            df_split = df[df['split'].str.strip().str.lower() == self.split.lower()].copy() # Use .copy() to avoid SettingWithCopyWarning
        else:
            df_split = df.copy()

        if df_split.empty:
            print(f"Warning: No samples found for split '{self.split}' in {csv_file_path}.")
            self.samples = []
            self.label_to_idx = {}
            self.idx_to_label = {}
            self.num_classes = 0
            return

        # Create label mapping if not provided
        if label_to_idx is None:
            unique_labels = sorted(df_split['label'].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx
        
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

        self.samples = []
        for _, row in df_split.iterrows():
            yt_id = row['youtube_id']
            start_sec = int(row['start_seconds'])
            label_str = row['label']
            
            # Format filename: e.g., azecEibOHVg_000020.mp4
            video_filename = f"{yt_id}_{start_sec:06d}.mp4"
            video_path = os.path.join(self.video_dir, video_filename)
            
            label_idx = self.label_to_idx.get(label_str, -1) # Use -1 for unknown labels if any
            if label_idx == -1:
                print(f"Warning: Label '{label_str}' not in provided label_to_idx map. Skipping sample.")
                continue

            self.samples.append((video_path, label_idx))
        
        print(f"VGGSoundDataset: Loaded {len(self.samples)} samples for split '{self.split}'. Found {self.num_classes} classes.")
        if not self.samples:
            print(f"Warning: No video paths were successfully constructed for split '{self.split}'. Check video_dir and CSV consistency.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label_idx = self.samples[idx]
        
        # For Stage 1 MAE, we primarily need the video_path.
        # The model will handle actual data loading and processing from the path.
        sample = {'video_path': video_path, 'label_idx': label_idx}

        if self.transform:
            # This part might be more relevant if dataset itself returned tensors
            sample = self.transform(sample) 
            
        return sample # Returns a dict: {'video_path': str, 'label_idx': int}


def vggsound_collate_fn(batch):
    """
    Collate function for VGGSoundDataset.
    `batch` is a list of dictionaries, each like {'video_path': str, 'label_idx': int}
    Returns a dictionary with batched data.
    """
    video_paths = [item['video_path'] for item in batch]
    label_indices = torch.tensor([item['label_idx'] for item in batch], dtype=torch.long)
    
    return {'video_paths_batch': video_paths, 'labels_batch': label_indices}


if __name__ == '__main__':
    print("Testing VGGSoundDataset and DataLoader...")

    # --- !!! IMPORTANT: USER ACTION REQUIRED !!! ---
    # Please replace these placeholder paths with the ACTUAL paths to your VGGSound data.
    home_dir = os.path.expanduser("~")
    placeholder_csv_path = os.path.join(home_dir, "848m", "T3_AV", "vggsound.csv") 
    placeholder_video_dir = os.path.join(home_dir, "848m", "VGGSound_dataset", "full_dataset_extracted", 
                                       "scratch", "shared", "beegfs", "hchen", "train_data", 
                                       "VGGSound_final", "video")
    # --- End User Action Required ---

    print(f"Using CSV path: {placeholder_csv_path}")
    print(f"Using video directory: {placeholder_video_dir}")

    if not os.path.exists(placeholder_csv_path):
        print(f"ERROR: CSV file not found at the placeholder path: {placeholder_csv_path}")
        print("Please update the placeholder_csv_path in the __main__ block of vggsound_dataset.py")
        exit()
    if not os.path.exists(placeholder_video_dir):
        print(f"ERROR: Video directory not found at the placeholder path: {placeholder_video_dir}")
        print("Please update the placeholder_video_dir in the __main__ block of vggsound_dataset.py")
        exit()

    # 1. Test Dataset instantiation with subset of actual CSV
    print("\n--- 1. Testing Dataset Initialization with first 10 rows --- ")
    temp_csv_for_test = "vggsound_head_test.csv"
    try:
        # Read first 10 data rows (adjust nrows if your CSV has a comment/header row you want to skip implicitly)
        # Assuming the header=None in VGGSoundDataset handles potential comment lines correctly.
        df_head = pd.read_csv(placeholder_csv_path, header=None, nrows=10, 
                              names=['youtube_id', 'start_seconds', 'label', 'split']) # Use same names as in Dataset
        df_head.to_csv(temp_csv_for_test, index=False, header=False) # Write without index or header
        print(f"Created temporary test CSV: {temp_csv_for_test} with {len(df_head)} rows.")

        # Test with temporary CSV (train split)
        print("\nTesting with temp CSV (train split):")
        # Use actual video dir, but dataset won't check existence in __init__
        train_dataset_head = VGGSoundDataset(csv_file_path=temp_csv_for_test, video_dir=placeholder_video_dir, split='train')
        print(f"Number of samples in head train dataset: {len(train_dataset_head)}")
        if len(train_dataset_head) > 0:
            print(f"First sample (head train): {train_dataset_head[0]}")
            print(f"Label mapping (head train): {train_dataset_head.label_to_idx}")
        else:
            print("Head train dataset is empty.")

        print("\nTesting with temp CSV (test split):")
        test_dataset_head = VGGSoundDataset(csv_file_path=temp_csv_for_test, video_dir=placeholder_video_dir, split='test')
        print(f"Number of samples in head test dataset: {len(test_dataset_head)}")
        if len(test_dataset_head) > 0:
            print(f"First sample (head test): {test_dataset_head[0]}") 
        
    except Exception as e:
        print(f"Error during subset testing: {e}")
    finally:
        if os.path.exists(temp_csv_for_test):
            os.remove(temp_csv_for_test) # Clean up temp csv
            print(f"Removed temporary test CSV: {temp_csv_for_test}")

    # # 2. Test with actual full paths (Optional - can be slow)
    # print("\n--- 2. Testing with actual full CSV (might be slow) ---")
    # # Re-running this part is optional but good for full check
    # try:
    #     train_dataset_actual = VGGSoundDataset(csv_file_path=placeholder_csv_path, video_dir=placeholder_video_dir, split='train')
    #     if len(train_dataset_actual) > 0:
    #         print(f"Number of samples in actual train dataset: {len(train_dataset_actual)}")
    #         print(f"Actual train dataset - First sample: {train_dataset_actual[0]}")
    #         print(f"Actual train dataset - Label mapping size: {train_dataset_actual.num_classes} classes")
    #     else:
    #         print("Actual train dataset is empty or failed to load. Check paths and CSV content.")
    # except Exception as e:
    #     print(f"Error loading full dataset: {e}")

    # 3. Test DataLoader with subset data
    print("\n--- 3. Testing DataLoader (with 10-row subset data) ---")
    # Need to recreate the dataset from the temp file for the loader test
    dataset_for_loader_test = None
    temp_csv_for_loader = "vggsound_head_loader_test.csv"
    try:
        df_head_loader = pd.read_csv(placeholder_csv_path, header=None, nrows=10, 
                                     names=['youtube_id', 'start_seconds', 'label', 'split'])
        df_head_loader.to_csv(temp_csv_for_loader, index=False, header=False)
        # Use train split for the loader test
        dataset_for_loader_test = VGGSoundDataset(csv_file_path=temp_csv_for_loader, video_dir=placeholder_video_dir, split='train')
        
        if len(dataset_for_loader_test) > 0:
            batch_size = 2
            data_loader = DataLoader(dataset_for_loader_test, batch_size=batch_size, shuffle=True, collate_fn=vggsound_collate_fn, num_workers=0)
            
            print(f"DataLoader initialized with batch size {batch_size}.")
            num_batches_to_show = 2
            print("Iterating through DataLoader batches...")
            for i, batch in enumerate(data_loader):
                if i >= num_batches_to_show:
                    break
                print(f"\nBatch {i+1}:")
                print(f"  Video paths: {batch['video_paths_batch']}")
                print(f"  Labels: {batch['labels_batch']}")
                print(f"  Labels shape: {batch['labels_batch'].shape}")
        else:
            print("Skipping DataLoader test as the subset dataset for loader is empty.")

    except Exception as e:
        print(f"Error during DataLoader test setup or iteration: {e}")
    finally:
        if os.path.exists(temp_csv_for_loader):
            os.remove(temp_csv_for_loader) # Clean up temp csv
            print(f"Removed temporary loader test CSV: {temp_csv_for_loader}")

    print("\nVGGSoundDataset and DataLoader test finished.")
    print("IMPORTANT: Ensure placeholder_csv_path and placeholder_video_dir are correctly set to your actual data paths to use this dataset effectively.") 