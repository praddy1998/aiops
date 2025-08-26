# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC ## 10 - Train and log to MLflow + Register

# COMMAND ----------
import mlflow, mlflow.sklearn
from pyspark.sql.functions import rand
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

mlflow.set_experiment("/Shared/experiments/taxi_fare")

pdf = (spark.table("mlops_demo.taxi.gold_training")
       .orderBy(rand())
       .limit(200_000)
       .toPandas())

X = pdf.drop(columns=["label"])
y = pdf["label"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

with mlflow.start_run():
    params = {"n_estimators":200, "random_state":42}
    model = RandomForestRegressor(**params).fit(X_train,y_train)

    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred, squared=False)

    mlflow.log_params(params)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model", input_example=X_test.head(5))
    mlflow.set_tag("dataset","nyc_taxi")
    run_id = mlflow.active_run().info.run_id

result = mlflow.register_model(f"runs:/{run_id}/model", "taxi_fare_rf")
print("Registered model:", result.name, "version:", result.version)
