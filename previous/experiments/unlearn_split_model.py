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

def unlearn_from_split(split_index):
    """
    Unlearn pre-selected data from a specific split and retrain the model
    
    Parameters:
    split_index -- Index of the split to unlearn from
    
    Returns:
    model -- The retrained model
    unlearned_indices -- Indices of unlearned data
    """
    # Load the split data
    X_train, X_val, y_train, y_val = load_split_data(split_index)
    
    # Load the pre-selected indices to unlearn
    unlearn_data_path = f"./data/unlearned/unlearn_data_split_{split_index}.npz"
    if not os.path.exists(unlearn_data_path):
        print(f"Error: Unlearning data not found at {unlearn_data_path}")
        print("Please run prepare_unlearning_data.py first")
        return None, None
        
    unlearn_data = np.load(unlearn_data_path)
    
    # Get the mask that identifies which rows to keep
    keep_mask = unlearn_data['rows_to_keep_mask']
    
    # Also get the local indices that were removed (for reference)
    local_indices_to_unlearn = unlearn_data['local_indices']
    n_unlearn = len(local_indices_to_unlearn)
    
    # Filter the data
    X_train_filtered = X_train[keep_mask]
    y_train_filtered = y_train[keep_mask]
    
    print(f"Original training data: {len(X_train)} samples")
    print(f"Unlearning {n_unlearn} samples ({n_unlearn/len(X_train)*100:.1f}%)")
    print(f"Remaining training data: {len(X_train_filtered)} samples")
    
    # Train a new model on the filtered data
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_filtered, y_train_filtered)
    
    # Evaluate on validation set
    y_pred_val = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Validation accuracy after unlearning: {val_accuracy:.4f}")
    
    return model, unlearn_data['original_indices']

def main():
    parser = argparse.ArgumentParser(description='Unlearn pre-selected data from a specific split and retrain the model')
    parser.add_argument('--split-index', type=int, default=3, help='Index of the split to unlearn from (default: 3)')
    args = parser.parse_args()
    
    # Chec if split data exists
    split_path = Path(f"./data/split_ensemble/split_{args.split_index}.csv")
    if not split_path.exists():
        print(f"Error: Split {args.split_index} not found. Run train_split_models.py first.")
        return
        
    # Check if unlearning data exists
    unlearn_data_path = Path(f"./data/unlearned/unlearn_data_split_{args.split_index}.npz")
    if not unlearn_data_path.exists():
        print(f"Error: Unlearning data not found. Run prepare_unlearning_data.py first with --split-index {args.split_index}")
        return
    
    # Load the ensemble models
    model_base_name = 'split_ensemble'
    models, scaler, encoders = load_ensemble_models(model_base_name)
    
    if not models:
        print("Error: No ensemble models found. Run train_split_models.py first.")
        return
    
    print(f"Loaded {len(models)} ensemble models")
    
    # Unlearn and retrain the model for the specified split
    updated_model, unlearned_indices = unlearn_from_split(args.split_index)
    
    if updated_model is None:
        return
        
    # Replace the old model with the new one
    models[args.split_index] = updated_model
    
    # Save the updated ensemble models
    model_base_name = 'split_ensemble_unlearned'
    save_ensemble_models(models, model_base_name, scaler, encoders)
    print(f"Updated ensemble saved with base name '{model_base_name}' in the models directory")
    
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