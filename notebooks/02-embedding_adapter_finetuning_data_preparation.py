# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # This Notebook will create synthetic question-context pairs to establish baseline retriever performance and prepare training data for adapter fine-tuning
# MAGIC
# MAGIC * Steps: 
# MAGIC   * Generate a synthetic question-context pair dataset
# MAGIC   * Calculate metrics (recall@k, precision@k and F1 score) for baseline retriever
# MAGIC   * Save the dataset for fine-tune embedding adapters
# MAGIC

# COMMAND ----------

# MAGIC %pip install --quiet databricks-sdk==0.24.0 mlflow==2.14.1 sentence-transformers==3.0.1 torch==2.3.0 transformers==4.40.1 accelerate==0.27.2
# MAGIC %pip install -U openai
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

import json, sentence_transformers, torch, yaml, os, gc, logging, time, requests, mlflow
import matplotlib.pyplot as plt
import pandas as pd

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound, ResourceDoesNotExist

from random import randint
from sentence_transformers import InputExample, losses, SentenceTransformer, SentencesDataset
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, TripletEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments 
from sentence_transformers.trainer import SentenceTransformerTrainer

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from typing import List, Callable

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Synthetic question-context pairs

# COMMAND ----------

import os
from openai import OpenAI

# Add your own secret and base_url
os.environ["OPENAI_API_KEY"] = API_KEY = dbutils.secrets.get("keyvault", "databricks_LLMs")
base_url = "https://your-base-url"
os.environ["BASE_URL"] = HOST = base_url

# COMMAND ----------

SYSTEM_PROMPT = """You are a helpful assistant. The user is a Tax Law expert and needs help."""

USER_PROMPT = """I need help generating a question that the following context can answer. It should be specific to the context and something that a Tax expert might find interesting.

<context>
{CONTEXT}
</context>

Provide your answer in single quotes as shown in the following list:

Examples:
    - 'This is a good example that provides an interesting question.'
    - "This is a bad example in double quotes"
    - 'This is a good example that a Tax expert would want to do research on'
    - This is a bad example without quotes. It also has extra commentary that is not what I asked for.
"""

LLM = "azure.gpt-4o-mini"

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, col
import pandas as pd

@pandas_udf("string")
def create_questions_udf(contexts: pd.Series) -> pd.Series:
    """A pandas UDF function to perform scoring on sample dataset
    """
    from openai import OpenAI

    oai = OpenAI(base_url=HOST, api_key=API_KEY)
    def create_question(context):

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(CONTEXT=context)}
        ]
        response = oai.chat.completions.create(messages=messages, 
                                               model=LLM, 
                                               temperature=0.0,
                                               max_tokens=250)

        return response.choices[0].message.content
    return pd.Series([create_question(context) for context in contexts])

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog};")
spark.sql(f"USE SCHEMA {schema};")

df_chunks_questions = spark.table(f"{table}_high_quality_chunks") \
  .withColumn("question", create_questions_udf(col("text")))
display(df_chunks_questions)

# COMMAND ----------

df_chunks_questions.write \
  .mode("overwrite") \
  .option("overWriteSchema", "true") \
  .saveAsTable(f'{table}_quality_chunk_question_context_pairs')

# COMMAND ----------

# MAGIC %md
# MAGIC # Define a custom metric for embedding with LLM as a judge

# COMMAND ----------

import mlflow
from mlflow.metrics.genai import make_genai_metric, EvaluationExample

# COMMAND ----------

#df_chunks_questions = spark.table("explore_chunks_quality_chunk_question_context_pairs")
df_chunks_questions_pd = df_chunks_questions.toPandas()

# COMMAND ----------

example_context1 = """## Place of Supply

Whether a supply is made in or outside Canada, and, if made in Canada, whether it is made in or outside a participating province (i.e., Nova Scotia, New Brunswick or Newfoundland and Labrador), affects the rate of tax that applies to that supply. Where a taxable (other than zero-rated) supply of property or a service is made in Canada, tax at the rate of 7% or 15% applies to that supply, depending on whether it is made in a participating province.

Generally, a supply of property or services made outside Canada is not subject to the GST/HST. However, the GST/HST may apply to property or services acquired by a resident outside Canada and subsequently imported into Canada. While tax is not collected at the border for taxable importations of intangible personal property or services, resident recipients of such supplies may be required to self-assess the tax.

The general rules for determining whether a supply is made in or outside Canada are set out in section 142 of the Act. However, section 143 of the Act overrides the general place of supply rules found in section 142 for supplies made by a non-resident who is not carrying on business in Canada and is not registered for GST/HST purposes at the time the supply is made.

Non-resident registration and carrying on business in Canada are explained in Non-Resident Registration."""

example_question1_s1 = """Example question 1: ‘When did the Canadian government introduce the maple leaf symbol on passports?’"""
example_question1_s2 = """Example question 2: ‘Are Canadian businesses required to collect GST/HST on goods sold within the country?’"""
example_question1_s3 = """Example question 3: ‘What role does section 142 play in determining whether a service provided by a non-resident is made in Canada?"""
example_question1_s4 = """Example question 4: ‘If a supplier from outside Canada provides digital services to customers in a participating province, which criteria under sections 142 and 143 might affect the applicable HST rate?’"""
example_question1_s5 = """Example question 5: ‘What compliance challenges could arise when a resident consumer self-assesses GST/HST on intangible services acquired from a non-resident, and how might sections 142 and 143 help determine liability?’"""

# COMMAND ----------

example1 = EvaluationExample(
    input=example_context1,
    output=(example_question1_s1),
    score=1,
    justification=(
        "This question is unrelated to determining the place of supply or GST/HST rules."
    )
)

example2 = EvaluationExample(
    input=example_context1,
    output=(example_question1_s2),
    score=2,
    justification=(
        "This question is loosely related to GST/HST but does not address the place of supply rules or distinction between provinces."
    )
)

example3 = EvaluationExample(
    input=example_context1,
    output=(example_question1_s3),
    score=3,
    justification=(
        "This question references the relevant section but mostly repeats the context without adding new perspectives."
    )
)

example4 = EvaluationExample(
    input=example_context1,
    output=(example_question1_s4),
    score=4,
    justification=(
        "This question introduces a focused scenario (digital services, participating province) and ties it to specific sections, expanding on the application of the rules."
    )
)

example5 = EvaluationExample(
    input=example_context1,
    output=(example_question1_s5),
    score=5,
    justification=(
        "This question is highly relevant and adds significant detail about compliance issues and self-assessment under the specified sections."
    )
)

# COMMAND ----------

example_context2 = """Example context 2: Voluntary Repayment of the CEWS

Principal Issues: Is a voluntary repayment of the CEWS deductible in computing a taxpayer’s profit from a business or property under subsection 9(1)? If so, in which taxation year would the deduction be permitted? If the CEWS was included in income under paragraph 12(1)(x) of the Act, would the voluntary repayment be deductible pursuant to paragraph 20(1)(hh) of the Act?

Position: Where the CEWS was originally included in computing a taxpayer’s profit from a business or property under subsection 9(1) of the Act, if the eligible entity were to complete the necessary steps to cancel the relevant CEWS application(s), a deduction would be permitted in computing a taxpayer’s profit from a business or property under subsection 9(1) of the Act, and would not be prohibited by the general limitation in 18(1)(a) of the Act, in the year there is a legal obligation to repay an amount. A deduction would also be permitted under paragraph 20(1)(hh) if an amount is repaid by a taxpayer in the year, pursuant to a legal obligation to repay all or part of a particular amount that was included under paragraph 12(1)(x) of the Act in computing income for the year or a preceding taxation year. A legal obligation, for purposes of subsection 9(1) and paragraph 20(1)(hh), would generally arise at the time the eligible entity completes the necessary steps to cancel the relevant CEWS application(s).

Reasons: See below.

XXXXXXXXXX        Aleksandra Bogdan CPA, CA
          2021-087629

October 4, 2021

Dear XXXXXXXXXX:

Re: Voluntary repayment of the Canada Emergency Wage Subsidy

This is in reply to your email in which you requested our views on a matter relating to the Canada Emergency Wage Subsidy (“CEWS”) under section 125.7 the Income Tax Act (“the Act”).

In the situation you described, an “eligible entity” (footnote 1) qualified and successfully applied for the CEWS for certain “qualifying periods” (footnote 2) in the spring of 2020. However, the eligible entity would now like to repay all the CEWS it received in 2020. Specifically, you would like to know whether a voluntary repayment of the CEWS, which was originally included in computing the eligible entity’s profit from a business or property under subsection 9(1) of the Act, is deductible under that subsection. Or, if the amount of the CEWS had been included in the taxpayer’s income under paragraph 12(1)(x) of the Act, would the voluntary repayment be deductible pursuant to paragraph 20(1)(hh) of the Act. Furthermore, you would like to know if a deduction is available, in which taxation year would the deduction be permitted.

Our comments

This technical interpretation provides general comments about the provisions of the Act and related legislation (where referenced). It does not confirm the income tax treatment of a particular situation involving a specific taxpayer but is intended to assist you in making that determination. The income tax treatment of particular transactions proposed by a specific taxpayer will only be confirmed by this Directorate in the context of an advance income tax ruling request submitted in the manner set out in Information Circular IC 70-6R11, Advance Income Tax Rulings and Technical Interpretations.

Deduction under subsection 9(1) of the Act

A taxpayer’s income for a taxation year from a business or property is determined under subsection 9(1) the Act. Subsection 9(1) provides that, subject to the provisions of Part I of the Act, a taxpayer’s income for a taxation year from a business or property is the taxpayer’s profit therefrom for the year. Paragraph 18(1)(a) of the Act, which is one of the general limitation provisions in Part I of the Act, provides that in computing the income of a taxpayer from a business or property, no deduction shall be made in respect of an outlay or expense except to the extent that it was made or incurred by the taxpayer for the purpose of gaining or producing income from the business or property.

The CRA has previously taken the position that the CEWS is generally required to be included in income under either subsection 9(1) of the Act or paragraph 12(1)(x) of the Act, depending on the manner in which a particular eligible entity determines its profit for income tax purposes. (footnote 3)

However, the requirement contained in paragraph 18(1)(a) of the Act that the outlay or expense be “made or incurred” contemplates the actual payment of the expense or a legal obligation to pay the expense in that particular taxation year. Generally, a taxpayer “incurs” an expense when there is a legal obligation to pay a sum of money.

That being said, where an eligible entity decides to voluntarily repay all the CEWS it received, it is our understanding that it would need to formally cancel its application(s) that had already been submitted by following the instructions provided on the Canada.ca website: Canada Emergency Wage Subsidy (CEWS) – Change or cancel your claim. (footnote 4) Upon cancellation of one or more CEWS applications, the eligible entity would be issued a Notice of Determination, pursuant to subsection 152(3.4) of the Act, which would provide the amount that must be repaid in respect of the cancelled applications.

Therefore, in the situation described, if the eligible entity were to complete the necessary steps to cancel the relevant CEWS application(s), a legal obligation would arise at the time of cancellation, in accordance with the Notice of Determination that would be issued, to repay an amount in respect of the cancelled application(s). This would result in an expense being incurred for the purposes of paragraph 18(1)(a) of the Act and provided the CEWS was initially included in computing the eligible entity’s profit from a business or property under subsection 9(1) of the Act, a deduction would be permitted in the year there is a legal obligation to repay an amount.

Deduction under paragraph 20(1)(hh) of the Act"""

example_question2_s1 = "Example question 1: Do taxpayers have to repay the CEWS if their revenue subsequently exceeds the initial qualification threshold?"
example_question2_s2 = "Example question 2: What is the purpose of the Canada Emergency Wage Subsidy for employers in Canada?"
example_question2_s3 = "Example question 3: How does a taxpayer normally include the CEWS in computing income, and can that amount be subsequently reversed?"
example_question2_s4 = "Example question 4: If an entity repays the CEWS in a different taxation year than when it was received, at what point does the legal obligation under subsection 9(1) arise for deduction purposes?"
example_question2_s5 = "Example question 5: What documentation or Notice of Determination must an eligible entity obtain to validate a voluntary repayment of CEWS under paragraph 20(1)(hh), and how does this impact their taxable income in subsequent years?"

# COMMAND ----------

example6 = EvaluationExample(
    input=example_context2,
    output=(example_question2_s1),
    score=1,
    justification=(
        "The question misconstrues the concept. It’s off-topic because the context specifically relates to voluntary repayments and legal obligations, not mandatory repayments if revenue changes."
    )
)

example7 = EvaluationExample(
    input=example_context2,
    output=(example_question2_s2),
    score=2,
    justification=(
        "This question is loosely relevant (CEWS) but does not address any details about voluntary repayment, deductibility, or the specific provisions cited."
    )
)

example8 = EvaluationExample(
    input=example_context2,
    output=(example_question2_s3),
    score=3,
    justification=(
        "It touches on the idea of including the CEWS in income and possibly reversing it, but it simply restates the general concept from the context without going deeper."
    )
)

example9 = EvaluationExample(
    input=example_context2,
    output=(example_question2_s4),
    score=4,
    justification=(
        "This question expands the discussion by focusing on timing and the creation of a legal obligation, adding nuance to the repayment scenario."
    )
)

example10 = EvaluationExample(
    input=example_context2,
    output=(example_question2_s5),
    score=4,
    justification=(
        "This question is highly relevant and specific, referencing the legal obligation, the paragraph 20(1)(hh) deduction, and how the repayment impacts future taxable income."
    )
)

# COMMAND ----------

LLM_judge = "gpt-4o-mini"

question_quality = make_genai_metric(
  name="GeneratedQuestionQuality",
  definition=(
      "Measures the quality of the LLM generated questions"),
  grading_prompt=(
      """Generated Question Quality: We will evaluate the generated questions based on their relevance to the key concepts in the provided context, their ability to expand upon the given information, and their potential to serve as effective training examples for an embedding model. Higher scores will be given to questions that clearly relate to the context, introduce new perspectives or applications, and strike a balance between broad and specific inquiries.
      - Score 1: The question is not related to the key concepts in the provided context or does not form a coherent question.
      - Score 2: The question is loosely related to the context but does not directly address the main concepts or ideas. It may be too vague or tangential to serve as an effective training example.
      - Score 3: The question addresses the key concepts from the context but does so in a way that largely repeats the provided information without adding new insights or applications. It may be overly specific or narrow in scope, limiting its usefulness for training.
      - Score 4: The question effectively relates to the key concepts in the context and introduces some new perspectives or applications. It may be a mix of broad and specific elements, providing a reasonable balance between relevance and novelty. The question could serve as a useful training example.
      - Score 5: The question skillfully captures the essence of the key concepts in the context while adding significant new insights, perspectives, or applications. It may be broad or specific, but it avoids repetition and offers a fresh angle on the topic. The question is highly relevant and informative, making it an excellent training example for an embedding model."""
  ),
  model=f"endpoints:/{LLM_judge}",
  examples=[example1, example3, example5, example7, example9],
  parameters={"temperature": 0.0},
  aggregations=["mean", "variance"],
  greater_is_better=True,
)

# COMMAND ----------

eval_pd = df_chunks_questions_pd.copy()
eval_pd["predictions"] = eval_pd["question"]
eval_pd["inputs"] = eval_pd["text"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate evaluation metrics with mlflow evaluate

# COMMAND ----------

with mlflow.start_run(run_name="embedding_eval_test") as run:
    question_eval = mlflow.evaluate(
        data=eval_pd,
        model_type="question-answering",
        predictions="predictions",
        extra_metrics=[question_quality],
      )

# COMMAND ----------

question_eval.metrics

# COMMAND ----------

import pandas as pd

eval_results_table = pd.DataFrame.from_dict(question_eval.tables["eval_results_table"])
eval_results_table.groupby("GeneratedQuestionQuality/v1/score").count()["question"]

# COMMAND ----------

spark.createDataFrame(eval_results_table).write \
  .mode("overwrite") \
  .option("overWriteSchema", "true") \
  .saveAsTable(f'explore_chunks_quality_chunk_question_context_pairs_evaluate_base')

# COMMAND ----------

# MAGIC %md 
# MAGIC # Generate Baseline Retriever Evaluation with Base Embedding Model's vector search index
# MAGIC
# MAGIC * Use generated questions to retrive chunks

# COMMAND ----------

from pyspark.sql.functions import col

df_chunks_questions_evaluate_base = spark.table(f'{table}_quality_chunk_question_context_pairs_evaluate_base')
df_chunks_questions_high_score = df_chunks_questions_evaluate_base.filter(col("GeneratedQuestionQuality/v1/score") >= 4)

# COMMAND ----------

train_df, eval_df = df_chunks_questions_high_score.randomSplit([0.8, 0.2], seed=42)
eval_pd = eval_df.toPandas()
eval_pd["id"] = eval_pd["id"].transform(lambda x: [x])
eval_pd.head()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import json

def get_relevant_documents(question:str, index_name:str, k:int = 3, filters:str = None, max_retries:int = 3) -> List[dict]:
    """
    This function searches through the supplied vector index name and returns relevant documents 
    """
    # print(question)
    w = WorkspaceClient()
    docs = w.vector_search_indexes.query_index(
        index_name=index_name,
        columns=["id", "text", "token_count"], # return these columns in the response
        filters_json=filters, # apply these filter statements in the query 
        num_results=k, # show k results in the response 
        query_text=question # query as text 
    )
    docs_pd = pd.DataFrame(docs.result.data_array)
    docs_pd.columns = [_c.name for _c in docs.manifest.columns]
    return json.loads(docs_pd.to_json(orient="records"))

# write a function that we will apply across the entire evaluation dataset to get the relevant document IDs 
def get_relevant_doc_ids(question : str) -> list[str]:
    docs = get_relevant_documents(question, index_name=f"{catalog}.{schema}.{table}_high_quality_chunks_vs_index", k=10)
    return [int(_x['id']) for _x in docs]

# test that it works for a single question
print(get_relevant_doc_ids(eval_pd.iloc[0]["question"]))

# # apply the function to the entire evaluation dataset
eval_pd["retrieved_docs"] = eval_pd["question"].transform(get_relevant_doc_ids)

# COMMAND ----------

with mlflow.start_run() as run:
    eval_results = mlflow.evaluate(
        data = eval_pd,
        model_type="retriever",
        targets="id",
        predictions="retrieved_docs",
        evaluators="default",
        extra_metrics=[mlflow.metrics.recall_at_k(i) for i in range(1,10,1)] + [mlflow.metrics.precision_at_k(i) for i in range(1,10,1)]
    )

# COMMAND ----------

eval_results.metrics

# COMMAND ----------

from matplotlib import pyplot as plt

plt.plot([eval_results.metrics[f"recall_at_{i}/mean"] for i in range(1,10,1)], label="recall")
plt.title("Recall at k")
plt.xlabel("k")
plt.legend()
plt.show()

# COMMAND ----------

plt.plot([eval_results.metrics[f"precision_at_{i}/mean"] for i in range(1,10,1)], label="precision")
plt.title("precision at k")
plt.xlabel("k")
plt.legend()
plt.show()

# COMMAND ----------

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)
  
plt.plot([f1_score(eval_results.metrics[f"recall_at_{i}/mean"], eval_results.metrics[f"precision_at_{i}/mean"]) 
          for i in range(1,10,1)], label="F1")
plt.title("F1 at k")
plt.xlabel("k")
plt.legend()
plt.show()

# COMMAND ----------

(
    train_df
        .write
        .mode("overwrite")
        .option("overwriteSchema","true")
        .saveAsTable(f"generated_questions_train")
)

(
    eval_df
        .write
        .mode("overwrite")
        .option("overwriteSchema","true")
    .saveAsTable(f"generated_questions_eval")
)
