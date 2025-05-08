import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_adult_data, load_adult_test, process_test_data, save_model
from utils.split_data import create_validation_split

def main():
    data_path = os.path.join('data', 'adult_income', 'adult.data')
    X, y, scaler, encoders = load_adult_data(data_path)
    
    # Create a validation split for model evaluation during training
    X_train, X_val, y_train, y_val = create_validation_split(X, y, val_size=0.2)
    
    print("Training logistic regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred_val = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print("Validation classification report:")
    print(classification_report(y_val, y_pred_val))
    
    # Save the model and preprocessing objects
    model_name = 'logistic_regression_original'
    save_model(model, model_name, scaler, encoders)
    print(f"Model saved as '{model_name}' in the models directory")
    
    # Load test data
    test_path = os.path.join('data', 'adult_income', 'adult.test')
    test_df = load_adult_test(test_path)
    X_test, y_test = process_test_data(test_df, scaler, encoders)
    
    # Evaluate on test
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Test classification report:")
    print(classification_report(y_test, y_pred_test))

if __name__ == "__main__":
    main()