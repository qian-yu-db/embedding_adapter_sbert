# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # This Notebook will create a vector search index based on the base embedding model
# MAGIC
# MAGIC * Use high-quality (score 2) chunks
# MAGIC * Use baseline embedding model to generate vector search index

# COMMAND ----------

# MAGIC %pip install --quiet databricks-sdk==0.24.0 mlflow==2.14.1
# MAGIC %pip install databricks-vectorsearch tiktoken
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text(name="target_catalog", label="Catalog", defaultValue="dev_catalog")
dbutils.widgets.text(name="target_schema", label="Schema", defaultValue="dev_schema")
dbutils.widgets.dropdown(name="source_table", label="Table", defaultValue="chunks", 
                         choices=["explore_chunks", "all_fulltext", "long_chunks", "short_chunks"])
dbutils.widgets.text(name="embedding_model_endpoint", label="Embedding Model Endpoint", defaultValue="snowflake-arctic-embed-m-long")

# COMMAND ----------

catalog = dbutils.widgets.get("target_catalog")
schema = dbutils.widgets.get("target_schema")
table = dbutils.widgets.get("source_table")
embedding_model_endpoint = dbutils.widgets.get("embedding_model_endpoint")
print(f"Parameters: target_catalog: {catalog}, target_schema: {schema}, source_table: {table}, embedding_model_endpoint: {embedding_model_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare Data

# COMMAND ----------

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound, ResourceDoesNotExist

from random import randint

# COMMAND ----------

table_name = f"{table}_quality_scored"

spark.sql(f"USE CATALOG {catalog};")
spark.sql(f"USE SCHEMA {schema};")

# COMMAND ----------

df_chunks = spark.sql(f"select * from {table_name} where chunk_quality = 2 order by rand(42) limit 1000")
print(df_chunks.count())
display(df_chunks)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf("int")
def token_count_udf(txts: pd.Series) -> pd.Series:

  import tiktoken
  encoding = tiktoken.get_encoding("cl100k_base")
  def token_count(txt):
      tokens = encoding.encode(txt, disallowed_special=())
      return len(tokens)
  return pd.Series([token_count(t) for t in txts])

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

df_chunks = df_chunks \
  .withColumn("token_count", token_count_udf(df_chunks.text)) \
  .select("id", "title", "url", "text", "token_count")
df_chunks = df_chunks.withColumn("source_id", df_chunks["id"])
df_chunks = df_chunks.withColumn("id", monotonically_increasing_id())
display(df_chunks)

# COMMAND ----------

df_chunks.write \
  .mode("overwrite") \
  .option("overWriteSchema", "true") \
  .saveAsTable(f"{table}_high_quality_chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Create a Vectro Search Index

# COMMAND ----------

# MAGIC %sql
# MAGIC -- To enable this table as the source of vector search index, we need to enable CDF
# MAGIC ALTER TABLE explore_chunks_high_quality_chunks SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
import time

vsc = VectorSearchClient()

def endpoint_exists(vsc, vs_endpoint_name):
  try:
    return vs_endpoint_name in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
  except Exception as e:
    raise e


def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    try:
      endpoint = vsc.get_endpoint(vs_endpoint_name)
    except Exception as e:
      #Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
      if "REQUEST_LIMIT_EXCEEDED" in str(e):
        print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. Please manually check your endpoint status")
        return
      else:
        raise e
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")


def index_exists(vsc, endpoint_name, index_full_name):
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False


def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, index_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 40 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME = "embedding_optimization"

if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

# source table
source_table_fullname = f"{catalog}.{schema}.{table}_high_quality_chunks"

# index name
vs_index_fullname = f"{catalog}.{schema}.{table}_high_quality_chunks_vs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_source_column='text',
    embedding_model_endpoint_name=embedding_model_endpoint
  )
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
else:
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")
