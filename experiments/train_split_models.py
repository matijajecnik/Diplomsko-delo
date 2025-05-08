import os
import sys
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_adult_data, load_adult_test, process_test_data, save_ensemble_models
from utils.split_data import split_data_for_training, load_split_data

def train_split_models(X, y, num_models=10, val_size=0.2, random_state=42):
    """
    Train multiple models on different splits of the data for ensemble learning
    
    Parameters:
    X -- Features data (only needed if splits don't exist yet)
    y -- Target data (only needed if splits don't exist yet)
    num_models -- Number of models to train
    val_size -- Validation set size for each split
    random_state -- Random seed for reproducibility
    
    Returns:
    models -- List of trained models
    val_accuracies -- List of validation accuracies
    """
    models = []
    val_accuracies = []
    
    # Check if splits exist
    split_dir = Path("./data/split_ensemble")
    if split_dir.exists() and len(list(split_dir.glob("split_*.csv"))) >= num_models:
        print(f"Loading {num_models} existing data splits...")
        split_data = []
        for i in range(num_models):
            try:
                X_train, X_val, y_train, y_val = load_split_data(i)
                split_data.append((X_train, X_val, y_train, y_val))
            except FileNotFoundError:
                print(f"Error: Split {i} not found. Make sure to provide X and y data to create splits.")
                return [], []
    else:
        if X is None or y is None:
            print("Error: Data splits don't exist and no X, y data provided to create them.")
            return [], []
        
        print(f"Creating and saving {num_models} data splits...")
        split_data = split_data_for_training(X, y, num_splits=num_models, test_size=val_size)
    
    print(f"Training {num_models} models for split unlearning ensemble...")
    
    for i, (X_train, X_val, y_train, y_val) in enumerate(split_data):
        print(f"Training model {i+1}/{num_models}...")
        model = LogisticRegression(max_iter=1000, random_state=random_state+i)
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred_val = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        val_accuracies.append(val_accuracy)
        
        print(f"  Model {i+1} validation accuracy: {val_accuracy:.4f}")
        
        models.append(model)
    
    return models, val_accuracies

def main():
    # Check if split data already exists ig
    split_dir = Path("./data/split_ensemble")
    data_path = os.path.join('data', 'adult_income', 'adult.data')
    
    if split_dir.exists() and len(list(split_dir.glob("split_*.csv"))) > 0:
        print("Found existing split data files...")
        # Get the scaler and encoders for test data processing
        _, _, scaler, encoders = load_adult_data(data_path)
        X, y = None, None
    else:
        # Load and process the adult.data file
        print("No existing splits found. Creating new splits from data...")
        X, y, scaler, encoders = load_adult_data(data_path)
    
    # Train an ensemble of models
    num_models = 10
    models, val_accuracies = train_split_models(
        X=X, y=y, 
        num_models=num_models
    )
    
    if not models:
        print("No models were trained. Exiting.")
        return
    
    mean_val_accuracy = np.mean(val_accuracies)
    print(f"\nMean validation accuracy across all models: {mean_val_accuracy:.4f}")
    
    # Save the ensemble models
    model_base_name = 'split_ensemble'
    save_ensemble_models(models, model_base_name, scaler, encoders)
    print(f"Ensemble of {num_models} models saved with base name '{model_base_name}' in the models directory")
    
    # Load test data and evaluate the ensemble
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
    print(f"Test accuracy of ensemble (majority vote): {test_accuracy:.4f}")
    print("Test classification report:")
    print(classification_report(y_test, majority_vote))

if __name__ == "__main__":
    main()