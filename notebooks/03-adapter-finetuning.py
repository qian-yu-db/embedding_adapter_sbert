# Databricks notebook source
# MAGIC %pip install -U databricks-sdk
# MAGIC %pip install -U sentence-transformers
# MAGIC %pip install -U mlflow
# MAGIC %pip install python-snappy==0.7.3
# MAGIC %pip install einops
# MAGIC %pip install torch==2.4.0 torchvision==0.19.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
import os
sys.path.append(os.path.abspath('.'))

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from adapters import LinearAdapter, TwoLayerAdapter
from fine_tuning_engine import SbertAdapterFinetuneEngine
from utils import *
from sentence_transformers import SentenceTransformer, models

# COMMAND ----------

# MAGIC %sh PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# COMMAND ----------

dbutils.widgets.text(name="target_catalog", label="Catalog", defaultValue="dev_catalog")
dbutils.widgets.text(name="target_schema", label="Schema", defaultValue="dev_schema")
dbutils.widgets.text(name="embedding_model", label="Embedding model", defaultValue="Snowflake/snowflake-arctic-embed-m-long")

# COMMAND ----------

target_catalog = dbutils.widgets.get("target_catalog")
target_schema = dbutils.widgets.get("target_schema")
embedding_model = dbutils.widgets.get("embedding_model")
print(f"target catalog: {target_catalog}, target schema: {target_schema}, embedding model: {embedding_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare Training / Eval Datasets

# COMMAND ----------

spark.sql(f"USE CATALOG {target_catalog};")
spark.sql(f"USE SCHEMA {target_schema};")

# COMMAND ----------

training_set = spark.table(f"generated_questions_train").select("id", "question", "text").toPandas()
eval_set = spark.table(f"generated_questions_eval").select("id", "question", "text").toPandas()

training_set['id'] = training_set['id'].astype(str)
eval_set['id'] = eval_set['id'].astype(str)

# COMMAND ----------

# MAGIC %md
# MAGIC # Set up SBERT Adapter Model for Fine-tuning: Linear Adapter

# COMMAND ----------

embedding_dim = SentenceTransformer(embedding_model, trust_remote_code=True).get_sentence_embedding_dimension()
adapter = LinearAdapter(embedding_dim=embedding_dim,
                        adapter_dim=embedding_dim,
                        bias=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## fine-tuning with a custom fine-tune engine
# MAGIC
# MAGIC - load training arguments
# MAGIC - create a fine-tune engine
# MAGIC - train
# MAGIC - save adapter

# COMMAND ----------

import yaml

with open("../config/training_args.yaml") as f:
    training_args = yaml.safe_load(f)

print(training_args)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define fine-tuning engine

# COMMAND ----------

engine = SbertAdapterFinetuneEngine(base_model_name=embedding_model,
                                    adapter_model=adapter,
                                    training_dataset=training_set,
                                    eval_dataset=eval_set,
                                    text_col_name="text",
                                    question_col_name="question",
                                    training_args=training_args)

# COMMAND ----------

engine.construct_custom_sbert()
engine.create_trainer()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Adapter

# COMMAND ----------

engine.train()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save trained adapter model

# COMMAND ----------

# Add your output dir
output_dir = '/Volumes/dev_catalog/dev_schema/my_volume/"
dbutils.fs.ls(output_dir)

# COMMAND ----------

engine.save_finetune_adapter(output_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Show the custom SBERT

# COMMAND ----------

custom_model = engine.model
print("=============================================")
print(f"transformer module: {custom_model[0]}")
print(f"{custom_model[0].auto_model}")
print("----------------------------------------------")
print(f"adapter module: {custom_model[1]}")
print("----------------------------------------------")
print(f"pooling module: {custom_model[2]}")
print("----------------------------------------------")
print(f"normalize module: {custom_model[3]}")
print("----------------------------------------------")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the custom SBERT to Unity Catalog

# COMMAND ----------

import mlflow

registered_ft_embedding_model_name = "snowflake-arctic-embed-m-long-linear-adapter"
data = "Look at my finetuned model"

# Log the model to unity catalog
mlflow.set_registry_uri("databricks-uc")

signature = mlflow.models.infer_signature(
    model_input=data,
    model_output=custom_model.encode(data),
)
model_uc_path = f"{target_catalog}.{target_schema}.{registered_ft_embedding_model_name}"

with mlflow.start_run() as run:

  # log the model to mlflow as a transformer with PT metadata
    logged_info = mlflow.sentence_transformers.log_model(
      model = custom_model,
      artifact_path="sbert_model",
      registered_model_name=model_uc_path,
      signature=signature
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log adapter.py file to model artifacts (This is a requirement for custom module for Sentence Transformer)

# COMMAND ----------

# update to your path
adapter_python_file = "/Workspace/Users/my_workspace_id/embedding_model_adapters/embedding_adapter_sbert/notebooks/adapters.py"

with mlflow.start_run(run_id=logged_info.run_id) as run:
    mlflow.log_artifact(local_path=adapter_python_file, artifact_path="sbert_model/model.sentence_transformer")

# COMMAND ----------

# MAGIC %md
# MAGIC # (Optional) Load adapter for reuse

# COMMAND ----------

# add your path
local_path = '/Volumes/dev_catalog/dev_schema/my_volume/my_path/model.sentence_transformer"
loaded_model = SentenceTransformer(model_name_or_path=local_path, trust_remote_code=True)

# COMMAND ----------

import json 

with open(f"{output_dir}/config.json", "r") as f:
    adapter_config = json.load(f)

adapter_config

# COMMAND ----------

adapter2 = LinearAdapter(embedding_dim=embedding_dim,
                         adapter_dim=embedding_dim,
                         bias=False)

# COMMAND ----------

from sentence_transformers import SentenceTransformer, models

adapter_module = adapter.load(output_dir)
base_model = SentenceTransformer(embedding_model, trust_remote_code=True)
transformer_module = base_model[0]
pooling_module = base_model[1]
normalize_module = base_model[2]

# COMMAND ----------

new_sbert_w_adapter = SentenceTransformer(modules=[transformer_module,
                                                   adapter_module,
                                                   pooling_module,
                                                   normalize_module])

# COMMAND ----------

embedding_example_new = new_sbert_w_adapter.encode("O Canada! Our home and native land!")

# COMMAND ----------

embedding_example_new1 = loaded_model.encode("O Canada! Our home and native land!")

# COMMAND ----------

embedding_example_new == embedding_example_new1

# COMMAND ----------

model_uc_path = f"{target_catalog}.{target_schema}.{registered_ft_embedding_model_name}_test"

with mlflow.start_run() as run:
  # log the model to mlflow as a transformer with PT metadata
    logged_info = mlflow.sentence_transformers.log_model(
      model = loaded_model,
      artifact_path="sbert_model",
      registered_model_name=model_uc_path,
      signature=signature
    )

# COMMAND ----------

with mlflow.start_run(run_id=logged_info.run_id) as run:
    mlflow.log_artifact(local_path=adapter_python_file, artifact_path="sbert_model/model.sentence_transformer")

# COMMAND ----------


