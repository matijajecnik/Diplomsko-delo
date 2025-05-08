import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_data_for_training(X, y, num_splits=10, test_size=0.2):
    """
    Split the data sequentially into multiple subsets for training ensemble models
    
    Parameters:
    X -- Features data
    y -- Target data
    num_splits -- Number of data splits to create
    test_size -- Fraction of data to use for validation in each split
    
    Returns:
    split_data -- List of tuples (X_train, X_val, y_train, y_val) for each split
    """
    split_data = []
    
    # Create directory if it doesnt exist
    split_dir = Path("./data/split_ensemble")
    split_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = X.shape[0]
    samples_per_split = n_samples // num_splits
    
    print(f"Total samples: {n_samples}, samples per split: {samples_per_split}")
    
    for i in range(num_splits):
        # Calculate start and end indices for this split
        start_idx = i * samples_per_split
        end_idx = (i + 1) * samples_per_split if i < num_splits - 1 else n_samples
        
        # Get the data for this split
        X_split = X[start_idx:end_idx]
        y_split = y[start_idx:end_idx]
        
        # Create validation split
        # Take the last test_size portion of data
        split_point = int(len(X_split) * (1 - test_size))
        X_train = X_split[:split_point]
        X_val = X_split[split_point:]
        y_train = y_split[:split_point]
        y_val = y_split[split_point:]
        
        split_data.append((X_train, X_val, y_train, y_val))
        
        # Save the original indices for reference
        indices = np.arange(start_idx, end_idx)
        train_indices = indices[:len(X_train)]
        val_indices = indices[len(X_train):]
        
        # Save in CSV format
        df_train = pd.DataFrame(X_train)
        df_train['target'] = y_train
        df_train['original_index'] = train_indices
        
        df_val = pd.DataFrame(X_val)
        df_val['target'] = y_val
        df_val['original_index'] = val_indices
        
        df_split = pd.DataFrame({
            'partition': ['train'] * len(X_train) + ['val'] * len(X_val),
            'original_index': np.concatenate([train_indices, val_indices]),
            'target': np.concatenate([y_train, y_val])
        })
        
        # Add all the features
        for j in range(X_train.shape[1]):
            df_split[f'feature_{j}'] = np.concatenate([X_train[:, j], X_val[:, j]])
        
        # Save as CSV
        df_split.to_csv(f"./data/split_ensemble/split_{i}.csv", index=False)
        
        # Also save train/val separately (why not)
        df_train.to_csv(f"./data/split_ensemble/split_{i}_train.csv", index=False)
        df_val.to_csv(f"./data/split_ensemble/split_{i}_val.csv", index=False)
    
    print(f"Saved {num_splits} sequential training data splits in ./data/split_ensemble/")
        
    return split_data

def load_split_data(split_index):
    """
    Load a specific split data from disk
    
    Parameters:
    split_index -- Index of the split to load
    
    Returns:
    X_train, X_val, y_train, y_val -- Training and validation data for the split
    """
    split_path = f"./data/split_ensemble/split_{split_index}.csv"
    df = pd.read_csv(split_path)
    
    # Separate train and validation data
    df_train = df[df['partition'] == 'train']
    df_val = df[df['partition'] == 'val']
    
    # Extract features, target, and indices
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    X_train = df_train[feature_cols].values
    y_train = df_train['target'].values
    
    X_val = df_val[feature_cols].values
    y_val = df_val['target'].values
    
    return X_train, X_val, y_train, y_val

def create_validation_split(X, y, val_size=0.2, random_state=42):
    """
    Create a single validation split from the training data
    
    Parameters:
    X -- Features data
    y -- Target data
    val_size -- Fraction of data to use for validation
    random_state -- Random seed for reproducibility
    
    Returns:
    X_train -- Training features
    X_val -- Validation features
    y_train -- Training targets
    y_val -- Validation targets
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state
    )
    
    return X_train, X_val, y_train, y_val