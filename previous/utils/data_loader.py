import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

def load_adult_data(data_path):
    """
    Load the adult.data training file and return processed X and y data
    """
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race",
        "sex", "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
    ]
    df = pd.read_csv(data_path, names=column_names, na_values="?", skipinitialspace=True)
    df = df.dropna()
    
    # Convert categorical features to numeric
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    
    X = df.drop("income", axis=1)
    y = df["income"]
    
    # Create a scaler and fit it to the training data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Return the data, scaler, and label encoders for future use
    return X_scaled, y.values, scaler, label_encoders

def load_adult_test(test_path):
    """
    Load the adult.test file and return processed X and y data
    """
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race",
        "sex", "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
    ]
    # Skip first line (header) 
    df = pd.read_csv(test_path, names=column_names, na_values="?", skipinitialspace=True, skiprows=1)
    df = df.dropna()
    
    # The test data has "." at the end of income values (e.g. ">50K."), so remove it
    df["income"] = df["income"].str.replace(".", "")
    
    return df

def process_test_data(test_df, scaler, label_encoders):
    """
    Process the test data using the same transformations applied to training data
    """
    test_df_copy = test_df.copy()
    
    # Apply the same label encoding as the training data
    for col, le in label_encoders.items():
        if col in test_df_copy.columns:
            # Handle unseen labels that might be in test data but not training
            test_df_copy[col] = test_df_copy[col].map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    X_test = test_df_copy.drop("income", axis=1)
    y_test = test_df_copy["income"]
    
    # Apply the same scaling as the training data
    X_test_scaled = scaler.transform(X_test)
    
    return X_test_scaled, y_test.values

def save_model(model, model_name, scaler=None, encoders=None):
    """
    Save a trained model to the models directory
    
    Parameters:
    model -- The trained model to save
    model_name -- Name of the model (used as filename)
    scaler -- The StandardScaler used to preprocess the data
    encoders -- Dictionary of LabelEncoders used to preprocess categorical features
    """
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save the model
    model_path = models_dir / f"{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save preprocessing objects if provided
    if scaler is not None:
        scaler_path = models_dir / f"{model_name}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    if encoders is not None:
        encoders_path = models_dir / f"{model_name}_encoders.pkl"
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders, f)

def load_model(model_name):
    """
    Load a saved model from the models directory
    
    Parameters:
    model_name -- Name of the model to load
    
    Returns:
    model -- The loaded model
    scaler -- The StandardScaler used with this model (if available)
    encoders -- Dictionary of LabelEncoders used with this model (if available)
    """
    models_dir = Path("models")
    
    # Load the model
    model_path = models_dir / f"{model_name}.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Try to load scaler and encoders if they exist
    scaler = None
    scaler_path = models_dir / f"{model_name}_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    encoders = None
    encoders_path = models_dir / f"{model_name}_encoders.pkl"
    if encoders_path.exists():
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
    
    return model, scaler, encoders

def save_ensemble_models(models, base_name, scaler=None, encoders=None):
    """
    Save an ensemble of models for split unlearning
    
    Parameters:
    models -- List of trained models
    base_name -- Base name for the models
    scaler -- The StandardScaler used to preprocess the data
    encoders -- Dictionary of LabelEncoders used to preprocess categorical features
    """
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save preprocessing objects once
    if scaler is not None:
        scaler_path = models_dir / f"{base_name}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    if encoders is not None:
        encoders_path = models_dir / f"{base_name}_encoders.pkl"
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders, f)
    
    # Save each model in the ensemble
    for i, model in enumerate(models):
        model_path = models_dir / f"{base_name}_{i}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

def load_ensemble_models(base_name, num_models=10):
    """
    Load an ensemble of models for split unlearning
    
    Parameters:
    base_name -- Base name of the models to load
    num_models -- Number of models in the ensemble
    
    Returns:
    models -- List of loaded models
    scaler -- The StandardScaler used with these models (if available)
    encoders -- Dictionary of LabelEncoders used with these models (if available)
    """
    models_dir = Path("models")
    models = []
    
    # Load each model in the ensemble
    for i in range(num_models):
        model_path = models_dir / f"{base_name}_{i}.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                models.append(model)
    
    # Load scaler and encoders if they exist
    scaler = None
    scaler_path = models_dir / f"{base_name}_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    encoders = None
    encoders_path = models_dir / f"{base_name}_encoders.pkl"
    if encoders_path.exists():
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
    
    return models, scaler, encoders