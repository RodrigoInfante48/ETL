"""
spark_ml.py — Machine learning with PySpark MLlib on economic data.

Models:
  1. K-Means clustering — group countries by economic profile
  2. Linear Regression — predict GDP per capita from macro indicators
  3. Feature importance — which indicators drive economic health score

All models are trained on the intermediate DataFrame from spark_transform.py.
"""

import os
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import (
    ClusteringEvaluator,
    RegressionEvaluator,
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from config import ML_MODELS_DIR, CSV_DIR


# ── Helpers ───────────────────────────────────────────────────────────────────

def _drop_nulls_for_ml(df: DataFrame, feature_cols: list[str], label_col: str | None = None) -> DataFrame:
    """Drop rows with nulls in the columns used for ML."""
    cols_to_check = feature_cols + ([label_col] if label_col else [])
    return df.dropna(subset=cols_to_check)


# ── 1. K-Means Country Clustering ─────────────────────────────────────────────

def cluster_countries_kmeans(df: DataFrame, k: int = 3) -> tuple[DataFrame, float]:
    """
    Cluster countries into `k` economic profiles using their average indicators.

    Groups countries by: avg_gdp_per_capita, avg_inflation, avg_unemployment, avg_health_score.
    Returns the cluster-annotated DataFrame and the silhouette score.
    """
    print(f"🤖 [K-Means] Clustering countries into {k} groups…")

    feature_cols = ["avg_gdp_per_capita", "avg_inflation", "avg_unemployment", "avg_health_score"]

    # Aggregate to country-level averages first (one row per country)
    country_avg = (
        df
        .groupBy("country", "is_latam")
        .agg(
            F.round(F.avg("gdp_per_capita"), 2).alias("avg_gdp_per_capita"),
            F.round(F.avg("inflation"), 2).alias("avg_inflation"),
            F.round(F.avg("unemployment_rate"), 2).alias("avg_unemployment"),
            F.round(F.avg("economic_health_score"), 1).alias("avg_health_score"),
        )
    )
    country_avg = _drop_nulls_for_ml(country_avg, feature_cols)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
    kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k, seed=42, maxIter=100)

    pipeline = Pipeline(stages=[assembler, scaler, kmeans])
    model = pipeline.fit(country_avg)
    clustered = model.transform(country_avg)

    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster", metricName="silhouette")
    silhouette = evaluator.evaluate(clustered)

    result = clustered.select(
        "country", "is_latam", "cluster",
        "avg_gdp_per_capita", "avg_inflation", "avg_unemployment", "avg_health_score",
    ).orderBy("cluster", "country")

    print(f"✅ [K-Means] Silhouette score: {silhouette:.4f}")
    result.show(truncate=False)

    # Cluster profiles
    print("\n📊 Cluster centroids (in scaled space):")
    km_model = model.stages[-1]
    for i, center in enumerate(km_model.clusterCenters()):
        print(f"   Cluster {i}: {[round(v, 2) for v in center]}")

    return result, silhouette


# ── 2. Linear Regression — GDP per capita prediction ──────────────────────────

def predict_gdp_per_capita(df: DataFrame) -> tuple[DataFrame, dict]:
    """
    Train a Linear Regression model to predict GDP per capita from:
      - year (captures time trend)
      - inflation
      - unemployment_rate
      - economic_health_score (derived feature)

    Returns predictions DataFrame and evaluation metrics dict.
    """
    print("🤖 [Linear Regression] Training GDP per capita predictor…")

    feature_cols = ["year", "inflation", "unemployment_rate", "economic_health_score"]
    label_col = "gdp_per_capita"

    ml_df = _drop_nulls_for_ml(df, feature_cols, label_col)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
    lr = LinearRegression(
        featuresCol="scaled_features",
        labelCol=label_col,
        predictionCol="predicted_gdp_per_capita",
        maxIter=100,
        regParam=0.1,    # L2 regularisation
        elasticNetParam=0.0,
    )

    pipeline = Pipeline(stages=[assembler, scaler, lr])

    # 80/20 train-test split (stratified by country would be ideal but simple split suffices)
    train, test = ml_df.randomSplit([0.8, 0.2], seed=42)
    print(f"   Train: {train.count()} rows | Test: {test.count()} rows")

    model = pipeline.fit(train)
    predictions = model.transform(test)

    evaluator_rmse = RegressionEvaluator(
        labelCol=label_col, predictionCol="predicted_gdp_per_capita", metricName="rmse"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol=label_col, predictionCol="predicted_gdp_per_capita", metricName="r2"
    )
    evaluator_mae = RegressionEvaluator(
        labelCol=label_col, predictionCol="predicted_gdp_per_capita", metricName="mae"
    )

    metrics = {
        "rmse": round(evaluator_rmse.evaluate(predictions), 2),
        "r2":   round(evaluator_r2.evaluate(predictions), 4),
        "mae":  round(evaluator_mae.evaluate(predictions), 2),
    }

    print(f"✅ [Linear Regression] RMSE={metrics['rmse']:,.2f}  R²={metrics['r2']}  MAE={metrics['mae']:,.2f}")

    # Coefficients — which features matter most?
    lr_model = model.stages[-1]
    print("\n📊 Coefficients (in scaled space):")
    for col, coef in zip(feature_cols, lr_model.coefficients):
        print(f"   {col:<30} {coef:+.4f}")
    print(f"   {'Intercept':<30} {lr_model.intercept:+.4f}")

    result = predictions.select(
        "country", "year", label_col, "predicted_gdp_per_capita",
        F.round(
            F.abs(F.col(label_col) - F.col("predicted_gdp_per_capita")),
            2
        ).alias("abs_error")
    ).orderBy(F.desc("abs_error"))

    return result, metrics


# ── 3. Feature importance via correlation ─────────────────────────────────────

def feature_importance_analysis(df: DataFrame) -> DataFrame:
    """
    Compute Pearson correlation between each macro indicator and economic_health_score.
    Quick proxy for feature importance without training a tree-based model.
    """
    print("📊 [Feature Importance] Correlating indicators with health score…")

    indicators = ["gdp_per_capita", "inflation", "unemployment_rate", "gdp_yoy_growth_pct"]
    target = "economic_health_score"

    clean = _drop_nulls_for_ml(df, indicators + [target])

    rows = []
    for ind in indicators:
        corr = clean.stat.corr(ind, target)
        rows.append((ind, round(corr, 4), round(abs(corr), 4)))

    result = df.sparkSession.createDataFrame(
        rows, ["indicator", "pearson_correlation", "abs_correlation"]
    ).orderBy(F.desc("abs_correlation"))

    print("\n📊 Correlation with economic_health_score:")
    result.show(truncate=False)
    return result


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_ml(intermediate_df: DataFrame) -> dict:
    """Run all ML experiments and return results."""
    os.makedirs(ML_MODELS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("🚀 Starting ML pipeline")
    print("=" * 60)

    cluster_df, silhouette = cluster_countries_kmeans(intermediate_df, k=3)
    predictions_df, lr_metrics = predict_gdp_per_capita(intermediate_df)
    importance_df = feature_importance_analysis(intermediate_df)

    # Persist results
    cluster_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{CSV_DIR}/ml_clusters")
    predictions_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{CSV_DIR}/ml_predictions")
    importance_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{CSV_DIR}/ml_feature_importance")

    print("\n✅ All ML outputs saved to CSV")
    return {
        "clusters": cluster_df,
        "silhouette": silhouette,
        "predictions": predictions_df,
        "lr_metrics": lr_metrics,
        "feature_importance": importance_df,
    }
