import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse
from functools import reduce

from pyspark.sql.functions import when, col, regexp_replace, length, trim,lower
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_clickstream(snapshot_date_str, bronze_lms_directory, silver_clickstream_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    # connect to bronze table
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    column_type_map = {
        "fe_1": IntegerType(),
        "fe_2": IntegerType(),
        "fe_3": IntegerType(),
        "fe_4": IntegerType(),
        "fe_5": IntegerType(),
        "fe_6": IntegerType(),
        "fe_7": IntegerType(),
        "fe_8": IntegerType(),
        "fe_9": IntegerType(),
        "fe_10": IntegerType(),
        "fe_11": IntegerType(),
        "fe_12": IntegerType(),
        "fe_13": IntegerType(),
        "fe_14": IntegerType(),
        "fe_15": IntegerType(),
        "fe_16": IntegerType(),
        "fe_17": IntegerType(),
        "fe_18": IntegerType(),
        "fe_19": IntegerType(),
        "fe_20": IntegerType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    for col_name, dtype in column_type_map.items():
        df = df.withColumn(col_name, df[col_name].cast(dtype))
    for i in range(1, 21):
        df = df.withColumn(f"fe_{i}", F.when(col(f"fe_{i}") < 0, 0).otherwise(col(f"fe_{i}")))
   
    df = df.withColumn("total_clicks", reduce(lambda a, b: a + b, [col(f"fe_{i}") for i in range(1, 21)]))
    df = df.withColumn("is_active", F.when(col("total_clicks") > 0, 1).otherwise(0))
  # save silver table - IRL connect to database to write
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_attributes(snapshot_date_str, bronze_lms_directory, silver_attributes_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    # connect to bronze table
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),  
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for col_name, dtype in column_type_map.items():
        df = df.withColumn(col_name, df[col_name].cast(dtype))


    avg_age = df.selectExpr("avg(Age)").collect()[0][0]
    df = df.withColumn("Age_clean", regexp_replace(col("Age").cast("string"), "[^0-9]", ""))
    df = df.withColumn("Age", when(col("Age_clean") != "", col("Age_clean").cast("int")).otherwise(None)).drop("Age_clean")
    df = df.withColumn("Age", when((col("Age") < 0) | (col("Age") > 100), None).otherwise(col("Age")))
    df = df.withColumn("Age", when(col("Age").isNull(), avg_age).otherwise(col("Age")))
    df = df.withColumn("Occupation", lower(trim(col("Occupation"))))
    df = df.withColumn("Occupation", when(col("Occupation").rlike("^_+$"), None).otherwise(col("Occupation")))
    df = df.withColumn("SSN_clean", regexp_replace(col("SSN"), "[^0-9]", ""))
    df = df.withColumn("SSN", when(length(col("SSN_clean")) > 0, col("SSN_clean")).otherwise(None)).drop("SSN_clean")

   # save silver table - IRL connect to database to write
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_financials(snapshot_date_str, bronze_lms_directory, silver_financials_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    # connect to bronze table
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    df = df.withColumn("Annual_Income", regexp_replace(col("Annual_Income").cast("string"), "[^0-9.]", "").cast("float"))
    df = df.withColumn("Num_of_Loan", regexp_replace(col("Num_of_Loan").cast("string"), "[^0-9]", "").cast("int"))
    df = df.withColumn("Num_Bank_Accounts", when(col("Num_Bank_Accounts") < 0, 0).otherwise(col("Num_Bank_Accounts")))
    df = df.withColumn("Type_of_Loan", when(trim(col("Type_of_Loan")) == "", None).otherwise(col("Type_of_Loan")))
    df = df.withColumn("Type_of_Loan", when(col("Type_of_Loan").rlike("^_+$"), None).otherwise(col("Type_of_Loan")))
    df = df.withColumn("Delay_from_due_date", when(col("Delay_from_due_date") < 0, 0).otherwise(col("Delay_from_due_date").cast("int")))
    df = df.withColumn("Num_of_Delayed_Payment", when(col("Num_of_Delayed_Payment") < 0, 0).otherwise(col("Num_of_Delayed_Payment").cast("int")))
    df = df.withColumn("Changed_Credit_Limit", col("Changed_Credit_Limit").cast("float"))
    df = df.withColumn("Changed_Credit_Limit", when(col("Changed_Credit_Limit") < 0, 0).otherwise(col("Changed_Credit_Limit")))
    df = df.withColumn("Credit_Mix", when(col("Credit_Mix").rlike("^_+$"), None).otherwise(col("Credit_Mix")))
    df = df.withColumn("Outstanding_Debt", regexp_replace(col("Outstanding_Debt").cast("string"), "[^0-9.]", "").cast("float"))
    df = df.withColumn("Amount_invested_monthly", regexp_replace(col("Amount_invested_monthly").cast("string"), "[^0-9.]", "").cast("float"))
    df = df.withColumn("Payment_Behaviour", regexp_replace(col("Payment_Behaviour"), "[^a-zA-Z0-9 ]", ""))
    df = df.withColumn("Payment_Behaviour", when(trim(col("Payment_Behaviour")) == "", None).otherwise(col("Payment_Behaviour")))
    df = df.withColumn("Annual_Income", when(col("Annual_Income") > 1_000_000, None).otherwise(col("Annual_Income")))
    df = df.withColumn("Num_of_Loan", when(col("Num_of_Loan") > 20, None).otherwise(col("Num_of_Loan")))
    df = df.withColumn("Num_of_Delayed_Payment", when(col("Num_of_Delayed_Payment") > 50, None).otherwise(col("Num_of_Delayed_Payment")))
    df = df.withColumn("Payment_Behaviour", when(col("Payment_Behaviour").rlike("^[0-9]+$") | (length(col("Payment_Behaviour")) > 30), None).otherwise(col("Payment_Behaviour")))

    column_type_map = {
        "Customer_ID": StringType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": IntegerType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for col_name, dtype in column_type_map.items():
        df = df.withColumn(col_name, df[col_name].cast(dtype))

   # save silver table - IRL connect to database to write
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df