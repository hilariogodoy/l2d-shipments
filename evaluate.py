import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model(model_name, model, X_test, y_test, threshold=0.30, cat_features=None):
    print(f"\n{'='*40}")
    print(f"--- Evaluating {model_name} ---")
    
    # For LightGBM and XGBoost, we need to pass categorical columns as pd.Categorical
    X_test_eval = X_test.copy()
    if model_name in ['LightGBM', 'XGBoost'] and cat_features is not None:
        for col in cat_features:
            X_test_eval[col] = X_test_eval[col].astype('category')
            
    # Get probabilities for class 1 (Delayed)
    y_prob = model.predict_proba(X_test_eval)[:, 1]
    
    # Apply custom threshold focusing on Recall
    y_pred = (y_prob >= threshold).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {auc:.4f}")
    
    print(f"\nClassification Report (Custom Threshold: {threshold}):")
    # Emphasize that recall of class 1 is what we optimize for
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print("                 Predicted On-Time (0) | Predicted Delayed (1)")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Actual On-Time (0) |        {cm[0,0]}          |        {cm[0,1]}")
    print(f"Actual Delayed (1) |        {cm[1,0]}          |        {cm[1,1]}")
    
    return y_prob, y_pred, cm, auc

def plot_feature_importance(model, model_name, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):
        importances = model.get_feature_importance()
    else:
        print(f"Cannot extract feature importance for {model_name}")
        return
        
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp, x='Importance', y='Feature', palette='mako')
    plt.title(f"Top 15 Feature Importances - {model_name}")
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig(f"models/{model_name.lower()}_feature_importance.png")
    plt.close()

def main():
    print("Loading test data...")
    if not os.path.exists("data/X_test.parquet"):
       raise FileNotFoundError("Test data not found. Please run train.py first.")
       
    X_test = pd.read_parquet("data/X_test.parquet")
    y_test = pd.read_parquet("data/y_test.parquet")['IS_DELAYED']
    cat_features = joblib.load("models/cat_features.pkl")
    
    models_to_eval = ['CatBoost', 'LightGBM', 'XGBoost']
    
    # Using customized threshold optimized for Recall of 1
    threshold = 0.30
    
    print(f"Evaluating models prioritizing RECALL over PRECISION:")
    print(f"Evaluating predictions based on customized threshold = {threshold}")
    
    for name in models_to_eval:
        try:
            model = joblib.load(f"models/{name.lower()}_model.pkl")
            evaluate_model(name, model, X_test, y_test, threshold, cat_features)
            plot_feature_importance(model, name, X_test.columns)
        except Exception as e:
            print(f"Could not evaluate {name}: {e}")

if __name__ == "__main__":
    main()
