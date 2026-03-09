import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the logistics shipment data.
    """
    df_clean = df.copy()
    
    numeric_cols = [
        'PARCEL_LENGTH_OPC', 'PARCEL_WIDTH_OPC', 'PARCEL_HEIGHT_OPC', 
        'PARCEL_WEIGHT_OPC', 'FIRST_MILE_TRANSIT_TIME_BD'
    ]
    # Convert specific columns to numeric
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Fill numeric NaNs with the group median based on X_3PL_NAME
    if 'X_3PL_NAME' in df_clean.columns:
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean.groupby('X_3PL_NAME')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Fallback if group median is still NaN
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    else:
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                
    # Fill categorical NaNs with 'No Data' and cast as string
    cat_cols = df_clean.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    expected_cat_cols = ['X_3PL_NAME', 'DELIVERY_PARTNER_CARRIER', 'LABEL_PRINT_ON_WEEKEND', 
                         'ORIGIN_PROCESSING_CENTER', 'PASSPORT_INVOICE_SERVICE_NAME', 
                         'DESTINATION_COUNTRY', 'DESTINATION_STATE', 'ROUTE_ID', 
                         'ORIGIN_COUNTRY', 'LABEL_PRINT_YEAR_WEEK_UTC']
    
    for col in set(cat_cols + expected_cat_cols):
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('No Data').astype(str)
            
    return df_clean

def train_stage_1_classifier(X_train, y_train, cat_features=None):
    """
    Train multiple classification models (Stage 1).
    Future Regression stage (Duration of delay) can be chained after this function.
    """
    os.makedirs("models", exist_ok=True)
    models = {}
    
    print("Checking for CatBoostClassifier...")
    if os.path.exists("models/catboost_model.pkl"):
        print("CatBoost model already exists. Loading from file...")
        cb = joblib.load("models/catboost_model.pkl")
    else:
        print("Training CatBoostClassifier...")
        cb = CatBoostClassifier(iterations=500, cat_features=cat_features, verbose=100)
        cb.fit(X_train, y_train)
        print("Saving CatBoostClassifier...")
        joblib.dump(cb, "models/catboost_model.pkl")
    models['CatBoost'] = cb
        
    print("Checking for LightGBM Classifier...")
    if os.path.exists("models/lightgbm_model.pkl"):
        print("LightGBM model already exists. Loading from file...")
        lgb = joblib.load("models/lightgbm_model.pkl")
    else:
        print("Training LightGBM Classifier...")
        # Prepare LightGBM (convert categorical columns to category type)
        X_train_lgb = X_train.copy()
        for col in cat_features:
            X_train_lgb[col] = X_train_lgb[col].astype('category')
            
        lgb = LGBMClassifier(n_estimators=500, random_state=42)
        lgb.fit(X_train_lgb, y_train)
        print("Saving LightGBM Classifier...")
        joblib.dump(lgb, "models/lightgbm_model.pkl")
    models['LightGBM'] = lgb
    
    print("Checking for XGBoost Classifier...")
    if os.path.exists("models/xgboost_model.pkl"):
        print("XGBoost model already exists. Loading from file...")
        xgb = joblib.load("models/xgboost_model.pkl")
    else:
        print("Training XGBoost Classifier...")
        # Prepare XGBoost categorical types (same as LightGBM)
        X_train_xgb = X_train.copy()
        for col in cat_features:
            X_train_xgb[col] = X_train_xgb[col].astype('category')
            
        xgb = XGBClassifier(
            n_estimators=500, 
            random_state=42, 
            enable_categorical=True, 
            tree_method='hist'
        )
        xgb.fit(X_train_xgb, y_train)
        print("Saving XGBoost Classifier...")
        joblib.dump(xgb, "models/xgboost_model.pkl")
    models['XGBoost'] = xgb
    
    return models

def main():
    print("Loading data...")
    if not os.path.exists("data/raw_shipments.parquet"):
        raise FileNotFoundError("data/raw_shipments.parquet not found. Run data_ingestion.py first.")
        
    df = pd.read_parquet("data/raw_shipments.parquet")
    
    print("Cleaning data...")
    df_clean = clean_data(df)
    
    target = 'IS_DELAYED'
    features = [col for col in df_clean.columns if col != target]
    
    # Identify categorical features dynamically
    cat_features = df_clean[features].select_dtypes(include=['object', 'string']).columns.tolist()
    
    X = df_clean[features]
    y = df_clean[target]
    
    print("Filtering data and making splits...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save splits for evaluation mode and app caching
    os.makedirs("data", exist_ok=True)
    X_test.to_parquet("data/X_test.parquet")
    pd.DataFrame(y_test).to_parquet("data/y_test.parquet")
    
    models = train_stage_1_classifier(X_train, y_train, cat_features)
    
    # Save a metadata file for categorical features
    joblib.dump(cat_features, "models/cat_features.pkl")
    print("Training phase completed!")

if __name__ == "__main__":
    main()
