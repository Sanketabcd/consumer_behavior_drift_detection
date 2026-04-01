import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def train_and_predict(baseline_df: pd.DataFrame, user_category: str, user_payment: str):
    """
    Trains a simple Random Forest Regressor on the baseline_df to predict
    Purchase_Amount based on Product_Category and Payment_Method.
    Returns the predicted amount and an expected standard deviation range.
    """
    df = baseline_df.dropna(subset=['Purchase_Amount', 'Product_Category', 'Payment_Method'])
    
    if len(df) < 5:
        # Fallback if there is not enough data to train
        return None, None
        
    X = df[['Product_Category', 'Payment_Method']]
    y = df['Purchase_Amount']
    
    # Preprocessor for categorical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Product_Category', 'Payment_Method'])
        ]
    )
    
    # Define a pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
    ])
    
    # Train model
    model.fit(X, y)
    
    # Predict
    input_data = pd.DataFrame([{
        'Product_Category': user_category,
        'Payment_Method': user_payment
    }])
    
    prediction = model.predict(input_data)[0]
    
    # Simple expected range: use standard deviation of tree predictions
    all_tree_preds = []
    # Since model is a pipeline, we need to transform the input first
    X_transformed = preprocessor.transform(input_data)
    for tree in model.named_steps['regressor'].estimators_:
        all_tree_preds.append(tree.predict(X_transformed)[0])
        
    std_dev = np.std(all_tree_preds)
    
    # Return predicted value and slightly padded range
    return float(prediction), float(std_dev)
