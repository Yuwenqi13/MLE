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
from pyspark.sql import Window
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_features_gold_table(snapshot_date_str, silver_loan_daily_directory,  silver_clickstream_directory,silver_attributes_directory, silver_financials_directory,gold_features_store_directory,spark):
    loan_path = silver_loan_daily_directory + f"silver_loan_daily_{snapshot_date_str.replace('-', '_')}.parquet"
    click_path = silver_clickstream_directory + f"silver_clickstream_{snapshot_date_str.replace('-', '_')}.parquet"
    attr_path = silver_attributes_directory + f"silver_attributes_{snapshot_date_str.replace('-', '_')}.parquet"
    financial_path = silver_financials_directory + f"silver_financials_{snapshot_date_str.replace('-', '_')}.parquet"
    
    loan_df = spark.read.parquet(loan_path)
    attr_df = spark.read.parquet(attr_path)
    click_df = spark.read.parquet(click_path)
    financial_df = spark.read.parquet(financial_path)
    
    df = loan_df.join(attr_df, ["Customer_ID", "snapshot_date"], "left")
    df = df.join(click_df, ["Customer_ID", "snapshot_date"], "left")
    df = df.join(financial_df, ["Customer_ID", "snapshot_date"], "left")

    window_spec = Window.partitionBy("Customer_ID", "snapshot_date").orderBy(F.desc("loan_id"))
    df = df.withColumn("rank", F.row_number().over(window_spec)) \
           .filter(F.col("rank") == 1) \
           .drop("rank")

    gold_path = gold_features_store_directory + f"gold_table_{snapshot_date_str.replace('-', '_')}.parquet"
    df.write.mode("overwrite").parquet(gold_path)
    print(f"gold table saved to {gold_path}")

    return df

