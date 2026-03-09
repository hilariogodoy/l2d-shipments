#%%
from _l2d_base import pull_from_snowflake
new_data = pull_from_snowflake('''   SELECT
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
            LABEL_PRINT_DATE_UTC between '2026-01-01' and '2026-01-31'
            AND IS_RETURN_SHIPMENT = FALSE
            AND IS_RETURN_TO_SENDER_SHIPMENT = FALSE
            AND (
                IS_DELIVERY_ATTEMPTED
                OR DELIVERY_EST_UTC_TS < CURRENT_DATE
            ) 
            LIMIT 10
            '''
)
#%%
from catboost import CatBoostClassifier
from_file = CatBoostClassifier()
model=from_file.load_model('l2d_shipments_2026-02-27_17-10-11.bin', format="cbm")
#%%
x_test_jan = new_data.drop('IS_DELAYED', axis=1)
y_test_jan = new_data['IS_DELAYED']
#%%
# Instead of model.predict(), use probability
probabilities = model.predict_proba(x_test_jan)[:, 1]

# Set a proactive threshold (e.g., 30% chance of delay is enough to flag)
preds_proactive = (probabilities > 0.30).astype(int)

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Generate predictions using your proactive threshold (e.g., 0.3)
preds = (model.predict_proba(x_test_jan)[:, 1] > 0.3).astype(int)

cm = confusion_matrix(y_test_jan, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistics Delay Confusion Matrix')
plt.show()

print(classification_report(y_test_jan, preds))

# %%

