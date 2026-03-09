#%%
import os
from dotenv import load_dotenv
import pandas as pd
import snowflake.connector

def pull_from_snowflake(sql):
    load_dotenv()

    USER = os.getenv("SNOWFLAKE_USER")
    PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
    ACCOUNT = f'{os.getenv("SNOWFLAKE_ORGANIZATION")}-{os.getenv("SNOWFLAKE_ACCOUNT")}'
    WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")

    with snowflake.connector.connect(
        user=USER, password=PASSWORD, account=ACCOUNT, warehouse=WAREHOUSE
    ) as con:
        with con.cursor() as cur:
            cur.execute(sql)
            df = cur.fetch_pandas_all()
            
            # Ensure column names are uppercase strings (standardizing Snowflake output)
            df.columns = [col.upper() for col in df.columns]
            
    return df
def clean_data(df):
    fillna_cols_num = ['PARCEL_LENGTH_OPC', 'PARCEL_WIDTH_OPC', 'PARCEL_HEIGHT_OPC', 'PARCEL_WEIGHT_OPC','FIRST_MILE_TRANSIT_TIME_BD']
    cat_features = ['X_3PL_NAME', 'ORIGIN_PROCESSING_CENTER',
            'PASSPORT_INVOICE_SERVICE_NAME', 'DELIVERY_PARTNER_CARRIER',
            'DESTINATION_COUNTRY', 'DESTINATION_STATE', 'ORIGIN_COUNTRY'
            ,'ROUTE_ID','LABEL_PRINT_YEAR_WEEK_UTC', 'LABEL_PRINT_ON_WEEKEND']
    # 1. Ensure numeric types (Fixes your TypeError)
    df[fillna_cols_num] = df[fillna_cols_num].apply(pd.to_numeric, errors='coerce')

    # 2. Fill numeric NaNs with group median
    df[fillna_cols_num] = df[fillna_cols_num].fillna(
        df.groupby('X_3PL_NAME')[fillna_cols_num].transform('median')
    )

    # 3. Handle categorical NaNs and cast to string
    df[cat_features] = df[cat_features].fillna('No Data').astype(str)
    return df
#%%

if __name__ == "__main__":
    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt
    from datetime import datetime
    import numpy as np

    df=pull_from_snowflake('''    SELECT
                X_3PL_NAME,
                ORIGIN_PROCESSING_CENTER,
                PASSPORT_INVOICE_SERVICE_NAME,
                DELIVERY_PARTNER_CARRIER,
                DESTINATION_COUNTRY,
                DESTINATION_STATE,
                ROUTE_ID,
                MILESTONE_LP_TO_MILESTONE_300_OR_GREATER_BD AS FIRST_MILE_TRANSIT_TIME_BD,
                ORIGIN_COUNTRY,
                LABEL_PRINT_YEAR_WEEK_UTC,
                CASE 
                    WHEN DAYOFWEEK(label_print_date_utc) IN (6, 0) THEN TRUE 
                    ELSE FALSE 
                END AS LABEL_PRINT_ON_WEEKEND,
                --CAST(label_print_utc_ts AS TIMESTAMP_NTZ) AS LABEL_PRINT_UTC_TS, // Model needs dates as timestamps_ntz
                --CAST(delivery_est_utc_ts AS TIMESTAMP_NTZ) AS DELIVERY_EST_UTC_TS,
                PARCEL_LENGTH_OPC,
                PARCEL_WIDTH_OPC,
                PARCEL_HEIGHT_OPC,
                PARCEL_WEIGHT_OPC,
                CASE
                    WHEN COALESCE(
                        milestone_greater_or_equal_631_utc_ts,
                        CURRENT_DATE()
                    ) > delivery_est_utc_ts THEN 1
                    ELSE 0
                END AS IS_DELAYED
            FROM
                PROD_ANALYTICS_DB.OPERATIONS_SCHEMA.NLT__OPERATIONS__SHIPMENT_DMT
            WHERE
                LABEL_PRINT_DATE_UTC BETWEEN '2024-01-01' AND '2025-12-31' -- 
                AND IS_RETURN_SHIPMENT = FALSE
                AND IS_RETURN_TO_SENDER_SHIPMENT = FALSE
                AND (
                    IS_DELIVERY_ATTEMPTED
                    OR DELIVERY_EST_UTC_TS < CURRENT_DATE
                ) 
                ''')
    #%%

    df = clean_data(df)

    #%%
    X = df.drop('IS_DELAYED', axis=1)
    y = df['IS_DELAYED']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # 3. Create a Pool object (Highly recommended for speed with 1M+ rows)
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)

    #%%

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',      # Focus on ranking delayed vs on-time
        random_seed=42,
        verbose=100,            # Print progress every 100 trees
        use_best_model=True,
        task_type="CPU"         # Use "GPU" if you have one; it's 10x faster for 1M rows
    )

    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

    #%%
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model.save_model(f'l2d_shipments_{now}.bin')

    #%%
    # Instead of model.predict(), use probability
    probabilities = model.predict_proba(X_test)[:, 1]

    # Set a proactive threshold (e.g., 30% chance of delay is enough to flag)
    preds_proactive = (probabilities > 0.30).astype(int)

    # Generate predictions using your proactive threshold (e.g., 0.3)
    preds = (model.predict_proba(X_test)[:, 1] > 0.3).astype(int)

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Logistics Delay Confusion Matrix')
    plt.show()

    print(classification_report(y_test, preds))

    #%%
    # Get feature importance from CatBoost
    feature_importance = model.get_feature_importance()
    feature_names = X_train.columns

    # 1. Get sorted indices based on importance
    indices = np.argsort(feature_importance)[::-1]

    # 2. Reorder feature names and importance based on sorted indices
    sorted_names = [feature_names[i] for i in indices]
    sorted_importance = [feature_importance[i] for i in indices]



    # Sort and plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importance, y=sorted_names)
    plt.title('What is driving our shipment delays?')
    plt.show()


# %%
