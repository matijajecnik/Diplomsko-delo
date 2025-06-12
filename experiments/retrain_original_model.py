import os
import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_adult_data, load_adult_test, process_test_data, save_model
from utils.data_loader import load_ensemble_models
from utils.split_data import create_validation_split

def retrain_original_model(split_index):
    """
    Retrain the original logistic regression model from scratch, but with the
    same data removed that was unlearned from the split ensemble model
    
    Parameters:
    split_index -- Index of the split whose data was unlearned
    
    Returns:
    model -- Retrained model
    scaler -- The data scaler used
    encoders -- The data encoders used
    """
    unlearn_data_path = f"./data/unlearned/unlearn_data_split_{split_index}.npz"
    if not os.path.exists(unlearn_data_path):
        print(f"Error: Unlearning data not found at {unlearn_data_path}")
        print("Please run prepare_unlearning_data.py first")
        return None, None, None
    
    unlearn_data = np.load(unlearn_data_path)
    original_indices_to_unlearn = unlearn_data['original_indices']
    
    data_path = os.path.join('data', 'adult_income', 'adult.data')
    X_full, y_full, scaler, encoders = load_adult_data(data_path)
    
    keep_mask = np.ones(len(X_full), dtype=bool)
    keep_mask[original_indices_to_unlearn] = False
    
    X_filtered = X_full[keep_mask]
    y_filtered = y_full[keep_mask]
    
    print(f"Original dataset size: {len(X_full)} samples")
    print(f"Removed {len(original_indices_to_unlearn)} samples that were unlearned from split {split_index}")
    print(f"Remaining dataset size: {len(X_filtered)} samples")
    
    X_train, X_val, y_train, y_val = create_validation_split(X_filtered, y_filtered, val_size=0.2, random_state=42)
    
    print("Training model on filtered data...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_val = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print("Validation classification report:")
    print(classification_report(y_val, y_pred_val))
    
    return model, scaler, encoders

def main():
    parser = argparse.ArgumentParser(description='Retrain original model from scratch with the same data removed as in the unlearned split')
    parser.add_argument('--split-index', type=int, default=3, help='Index of the split whose data was unlearned (default: 3)')
    args = parser.parse_args()
    
    unlearn_data_path = Path(f"./data/unlearned/unlearn_data_split_{args.split_index}.npz")
    if not unlearn_data_path.exists():
        print(f"Error: Unlearning data not found. Run prepare_unlearning_data.py first with --split-index {args.split_index}")
        return
    
    model, scaler, encoders = retrain_original_model(args.split_index)
    
    if model is None:
        return
    
    model_name = f'retrained_original_split_{args.split_index}'
    save_model(model, model_name, scaler, encoders)
    print(f"Retrained model saved as '{model_name}' in the models directory")
    
    # Load test data and evaluate the retrained model
    test_path = os.path.join('data', 'adult_income', 'adult.test')
    test_df = load_adult_test(test_path)
    X_test, y_test = process_test_data(test_df, scaler, encoders)
    
    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Test classification report:")
    print(classification_report(y_test, y_pred_test))
    
    # Compare with the unlearned ensemble model
    try:
        print("\nComparing with the unlearned ensemble model:")
        unlearned_ensemble_models, _, _ = load_ensemble_models('split_ensemble_unlearned')
        
        if unlearned_ensemble_models:
            # Perform ensemble voting on test set
            ensemble_predictions = []
            for model in unlearned_ensemble_models:
                predictions = model.predict(X_test)
                ensemble_predictions.append(predictions)
            
            # Take majority vote for each sample
            ensemble_predictions = np.array(ensemble_predictions)
            majority_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x)), 
                axis=0, 
                arr=ensemble_predictions
            )
            
            # Compare performances
            ensemble_accuracy = accuracy_score(y_test, majority_vote)
            print(f"Unlearned ensemble test accuracy: {ensemble_accuracy:.4f}")
            print(f"Retrained model test accuracy: {test_accuracy:.4f}")
            print(f"Difference: {(test_accuracy - ensemble_accuracy):.4f}")
        else:
            print("No unlearned ensemble models found. Run unlearn_split_model.py first.")
    except Exception as e:
        print(f"Error loading unlearned ensemble models: {e}")

if __name__ == "__main__":
    main()