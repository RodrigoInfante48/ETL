"""
SparkSession factory — single place to create/retrieve the Spark context.
"""

import os
from pyspark.sql import SparkSession
from config import SPARK_CONFIG, JDBC_JAR_PATH


def get_spark_session(app_name: str | None = None) -> SparkSession:
    """
    Return (or create) a SparkSession with the project defaults.

    If JDBC_JAR_PATH exists on disk it is added to the classpath automatically
    so that JDBC reads from PostgreSQL work without manual spark-submit flags.
    """
    name = app_name or SPARK_CONFIG["app_name"]

    builder = (
        SparkSession.builder
        .appName(name)
        .master(SPARK_CONFIG["master"])
        .config("spark.executor.memory", SPARK_CONFIG["executor_memory"])
        .config("spark.driver.memory", SPARK_CONFIG["driver_memory"])
        # Avoids Spark writing _SUCCESS files and hidden metadata files
        .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        # Show full column values in .show() — useful for long string columns
        .config("spark.sql.repl.eagerEval.truncate", "200")
    )

    if os.path.exists(JDBC_JAR_PATH):
        builder = builder.config("spark.jars", os.path.abspath(JDBC_JAR_PATH))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(SPARK_CONFIG["log_level"])
    return spark


def stop_spark(spark: SparkSession) -> None:
    """Gracefully shut down the SparkSession."""
    spark.stop()
    print("✅ SparkSession stopped.")
