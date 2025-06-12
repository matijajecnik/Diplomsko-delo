import os
import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_adult_data

def prepare_unlearning_data(split_index, percentage_to_unlearn=20, random_seed=42):
    """
    Prepare data for unlearning experiments by:
    1. Extracting a percentage of data from a specific split
    2. Identifying those records in the full dataset
    3. Saving information needed for unlearning experiments
    
    Parameters:
    split_index -- Index of the split to extract data from
    percentage_to_unlearn -- Percentage of data to extract from the split (default: 20%)
    random_seed -- Random seed for reproducibility
    
    Returns:
    unlearn_indices -- Dictionary with indices of data to unlearn
    """
    print(f"Preparing data for unlearning experiments...")
    
    np.random.seed(random_seed)
    
    split_path = f"./data/split_ensemble/split_{split_index}.csv"
    if not os.path.exists(split_path):
        print(f"Error: Split file {split_path} not found. Run train_split_models.py first.")
        return None
    
    df_split = pd.read_csv(split_path)
    df_train = df_split[df_split['partition'] == 'train']
    
    train_indices = df_train['original_index'].values
    n_samples = len(train_indices)
    n_unlearn = int(n_samples * percentage_to_unlearn / 100)
    
    print(f"Split {split_index} training data contains {n_samples} samples")
    print(f"Will extract {n_unlearn} samples ({percentage_to_unlearn}%) for unlearning")
    
    local_indices_to_unlearn = np.random.choice(n_samples, n_unlearn, replace=False)
    
    original_indices_to_unlearn = train_indices[local_indices_to_unlearn]
    
    rows_to_unlearn_in_split = np.zeros(n_samples, dtype=bool)
    rows_to_unlearn_in_split[local_indices_to_unlearn] = True
    
    unlearn_dir = Path("./data/unlearned")
    unlearn_dir.mkdir(parents=True, exist_ok=True)
    
    unlearn_data = {
        'split_index': split_index,
        'percentage_unlearned': percentage_to_unlearn,
        'local_indices': local_indices_to_unlearn,
        'original_indices': original_indices_to_unlearn,
        'rows_to_keep_mask': ~rows_to_unlearn_in_split
    }
    
    # Save unlearning data
    np.savez(
        f"./data/unlearned/unlearn_data_split_{split_index}.npz",
        **unlearn_data
    )
    np.save(f"./data/unlearned/unlearned_indices_split_{split_index}.npy", 
            original_indices_to_unlearn)
    
    data_path = os.path.join('data', 'adult_income', 'adult.data')
    X_full, y_full, _, _ = load_adult_data(data_path)
    
    X_to_unlearn = X_full[original_indices_to_unlearn]
    y_to_unlearn = y_full[original_indices_to_unlearn]
    
    unlearn_df = pd.DataFrame(X_to_unlearn)
    unlearn_df['target'] = y_to_unlearn
    unlearn_df['original_index'] = original_indices_to_unlearn
    unlearn_df.to_csv(f"./data/unlearned/unlearned_data_split_{split_index}.csv", index=False)
    
    print(f"Unlearning data prepared and saved in ./data/unlearned/")
    
    return unlearn_data

def main():
    parser = argparse.ArgumentParser(description='Prepare data for unlearning experiments')
    parser.add_argument('--split-index', type=int, default=3, 
                        help='Index of the split to extract data from (default: 3)')
    parser.add_argument('--percentage', type=float, default=20, 
                        help='Percentage of data to extract from the split (default: 20%)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed (default: 42)')
    args = parser.parse_args()
    
    prepare_unlearning_data(
        split_index=args.split_index,
        percentage_to_unlearn=args.percentage,
        random_seed=args.seed
    )

if __name__ == "__main__":
    main()