import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def pull_from_snowflake(sql: str) -> pd.DataFrame:
    """
    Connects to Snowflake, executes the query, fetches results as a Pandas DataFrame,
    and converts all column names to uppercase.
    """
    user = os.getenv("SNOWFLAKE_USER")
    password = os.getenv("SNOWFLAKE_PASSWORD")
    organization = os.getenv("SNOWFLAKE_ORGANIZATION")
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
    
    # Formatting the account as {ORGANIZATION}-{ACCOUNT}
    account_identifier = f"{organization}-{account}"
    
    conn = None
    try:
        conn = snowflake.connector.connect(
            user=user,
            password=password,
            account=account_identifier,
            warehouse=warehouse
        )
        
        # Execute query and fetch as Pandas DataFrame
        curr = conn.cursor()
        curr.execute(sql)
        df = curr.fetch_pandas_all()
        
        # Convert all column names to uppercase
        df.columns = [col.upper() for col in df.columns]
        
        return df
        
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    sql_query = """
    SELECT
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
    FROM PROD_ANALYTICS_DB.OPERATIONS_SCHEMA.NLT__OPERATIONS__SHIPMENT_DMT
    WHERE LABEL_PRINT_DATE_UTC BETWEEN '2024-01-01' AND '2025-12-31' 
        AND IS_RETURN_SHIPMENT = FALSE
        AND IS_RETURN_TO_SENDER_SHIPMENT = FALSE
        AND (IS_DELIVERY_ATTEMPTED OR DELIVERY_EST_UTC_TS < CURRENT_DATE)
    """
    
    print("Pulling data from Snowflake...")
    df = pull_from_snowflake(sql_query)
    
    # Save the resulting DataFrame locally
    os.makedirs("data", exist_ok=True)
    out_path = "data/raw_shipments.parquet"
    print(f"Saving data to {out_path} ...")
    df.to_parquet(out_path, engine="pyarrow")
    print("Data ingestion complete!")
