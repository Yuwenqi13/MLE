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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_feature
import utils.data_processing_silver_feature
import utils.data_processing_gold_feature
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")  

snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

bronze_lms_directory_loan = "datamart/bronze/lms_loan/"
bronze_lms_directory_clickstream = "datamart/bronze/lms_clickstream/"
bronze_lms_directory_attributes = "datamart/bronze/lms_attributes/"
bronze_lms_directory_financials = "datamart/bronze/lms_financials/"

for dir_path in [bronze_lms_directory_loan, bronze_lms_directory_clickstream, bronze_lms_directory_attributes, bronze_lms_directory_financials]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
for date_str in dates_str_lst:
    utils.data_processing_bronze_feature.process_bronze_loan(date_str, bronze_lms_directory_loan, spark)
    utils.data_processing_bronze_feature.process_bronze_clickstream(date_str, bronze_lms_directory_clickstream, spark)
    utils.data_processing_bronze_feature.process_bronze_attributes(date_str, bronze_lms_directory_attributes, spark)
    utils.data_processing_bronze_feature.process_bronze_financials(date_str, bronze_lms_directory_financials, spark)
silver_loan_daily_directory = "datamart/silver/loan_daily/"
silver_loan_directory_clickstream = "datamart/silver/loan_clickstream/"
silver_loan_directory_attributes = "datamart/silver/loan_attributes/"
silver_loan_directory_financials = "datamart/silver/loan_financials/"

if not all(os.path.exists(p) for p in [
    silver_loan_daily_directory,
    silver_loan_directory_clickstream,
    silver_loan_directory_attributes,
    silver_loan_directory_financials
]):
    for p in [
        silver_loan_daily_directory,
        silver_loan_directory_clickstream,
        silver_loan_directory_attributes,
        silver_loan_directory_financials
    ]:
        os.makedirs(p, exist_ok=True)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_feature.process_silver_table(date_str, bronze_lms_directory_loan, silver_loan_daily_directory, spark)
    utils.data_processing_silver_feature.process_silver_clickstream(date_str, bronze_lms_directory_clickstream, silver_loan_directory_clickstream, spark)
    utils.data_processing_silver_feature.process_silver_attributes(date_str, bronze_lms_directory_attributes, silver_loan_directory_attributes, spark)
    utils.data_processing_silver_feature.process_silver_financials(date_str, bronze_lms_directory_financials, silver_loan_directory_financials, spark)
    from pyspark.sql import functions as F

gold_feature_store_directory = "datamart/gold/feature_store/"
silver_loan_daily_directory = "datamart/silver/loan_daily/"
silver_loan_directory_clickstream = "datamart/silver/loan_clickstream/"
silver_loan_directory_attributes = "datamart/silver/loan_attributes/"
silver_loan_directory_financials = "datamart/silver/loan_financials/"

if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_feature.process_features_gold_table(date_str, silver_loan_daily_directory,  silver_loan_directory_clickstream,silver_loan_directory_attributes, silver_loan_directory_financials,gold_feature_store_directory,spark)

folder_path = gold_feature_store_directory
files_list = glob.glob(os.path.join(folder_path, '*')) 
df = spark.read.option("header", "true").parquet(*files_list)

print("row_count:", df.count())
df.show()

df = spark.read.option("header", "true").parquet(*files_list)
total_count = df.count()
null_ratio = {c: df.filter(F.col(c).isNull()).count() / total_count for c in df.columns}
keep_cols = [c for c, ratio in null_ratio.items() if ratio <= 0.7]
clean_df = df.select(*keep_cols)
print(f"After dropping columns with >70% nulls, remaining columns: {len(keep_cols)}")
clean_df.show(truncate=False)