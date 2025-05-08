import os
import sys
import numpy as np
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_adult_test, process_test_data
from utils.data_loader import load_ensemble_models, save_ensemble_models
from utils.split_data import load_split_data

def unlearn_from_split(split_index, percentage_to_unlearn=10, random_seed=42):
    """
    Unlearn a percentage of data from a specific split and retrain the model
    
    Parameters:
    split_index -- Index of the split to unlearn from
    percentage_to_unlearn -- Percentage of data to unlearn (default: 10%)
    random_seed -- Random seed for reproducibility
    
    Returns:
    model -- The retrained model
    unlearned_indices -- Indices of unlearned data
    """
    # Load the split data
    X_train, X_val, y_train, y_val = load_split_data(split_index)
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Number of samples to unlearn
    n_samples = len(X_train)
    n_unlearn = int(n_samples * percentage_to_unlearn / 100)
    
    # Randomly select rows to unlearnž
    unlearn_indices = np.random.choice(n_samples, n_unlearn, replace=False)
    
    keep_mask = np.ones(n_samples, dtype=bool)
    keep_mask[unlearn_indices] = False
    
    X_train_filtered = X_train[keep_mask]
    y_train_filtered = y_train[keep_mask]
    
    print(f"Original training data: {n_samples} samples")
    print(f"Unlearning {n_unlearn} samples ({percentage_to_unlearn}%)")
    print(f"Remaining training data: {len(X_train_filtered)} samples")
    
    # Train a new model on the filtered data
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_filtered, y_train_filtered)
    
    # Evaluate on validation set
    y_pred_val = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Validation accuracy after unlearning: {val_accuracy:.4f}")
    
    return model, unlearn_indices

def main():
    parser = argparse.ArgumentParser(description='Unlearn data from a specific split and retrain the model')
    parser.add_argument('--split-index', type=int, default=0, help='Index of the split to unlearn from (default: 0)')
    parser.add_argument('--percentage', type=float, default=10, help='Percentage of data to unlearn (default: 10%)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()
    
    # Chec if split data exists
    split_path = Path(f"./data/split_ensemble/split_{args.split_index}.csv")
    if not split_path.exists():
        print(f"Error: Split {args.split_index} not found. Run train_split_models.py first.")
        return
    
    model_base_name = 'split_ensemble'
    models, scaler, encoders = load_ensemble_models(model_base_name)
    
    if not models:
        print("Error: No ensemble models found. Run train_split_models.py first.")
        return
    
    print(f"Loaded {len(models)} ensemble models")
    
    # Unlearn and retrain the model for the index split
    updated_model, unlearned_indices = unlearn_from_split(
        args.split_index, 
        percentage_to_unlearn=args.percentage,
        random_seed=args.seed
    )
    
    # Replace the old model with new one
    models[args.split_index] = updated_model
    
    # Save the updated ensemble models
    model_base_name = 'split_ensemble_unlearned'
    save_ensemble_models(models, model_base_name, scaler, encoders)
    print(f"Updated ensemble saved with base name '{model_base_name}' in the models directory")
    
    # Save the unlearned data indices for reference
    unlearn_dir = Path("./data/unlearned")
    unlearn_dir.mkdir(parents=True, exist_ok=True)
    np.save(f"{unlearn_dir}/unlearned_indices_split_{args.split_index}.npy", unlearned_indices)
    
    # Evaluate the updated ensemble on the test data
    test_path = os.path.join('data', 'adult_income', 'adult.test')
    test_df = load_adult_test(test_path)
    X_test, y_test = process_test_data(test_df, scaler, encoders)
    
    # Perform ensemble voting on test set
    ensemble_predictions = []
    for model in models:
        predictions = model.predict(X_test)
        ensemble_predictions.append(predictions)
    
    # Take majority vote for each sample
    ensemble_predictions = np.array(ensemble_predictions)
    majority_vote = np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x)), 
        axis=0, 
        arr=ensemble_predictions
    )
    
    # Evaluate ensemble performance on test set
    test_accuracy = accuracy_score(y_test, majority_vote)
    print(f"Test accuracy of updated ensemble (majority vote): {test_accuracy:.4f}")
    print("Test classification report:")
    print(classification_report(y_test, majority_vote))

if __name__ == "__main__":
    main()