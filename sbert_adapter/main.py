from pathlib import Path

import mlflow
import pandas as pd
import yaml
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer

from adapters import *
from fine_tuning_engine import SbertAdapterFinetuneEngine

logging.basicConfig()
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def get_spark() -> SparkSession:
    """ Create a spark session with Databricks connect
    """
    try:
        from databricks.connect import DatabricksSession
        return DatabricksSession.builder.getOrCreate()
    except ImportError:
        return SparkSession.builder.getOrCreate()


def get_training_local(file_path):
    # read parquet file
    df = pd.read_parquet(file_path)
    logger.info(f"df samples: {df.columns}")

    # create train, eval set
    df_train_eval = df.loc[:, ['id', "text", "question"]].copy()
    df_train_eval['id'] = df_train_eval['id'].astype(str)
    training_df = df_train_eval.sample(frac=0.8, random_state=42)
    eval_df = df_train_eval.drop(training_df.index)
    logger.info(f"training set shape: {training_df.shape} | eval set shape {eval_df.shape}")

    return training_df, eval_df


def run_finetune():
    project_root = Path('../../')
    input_file_path = project_root / 'data/question_context_pairs.parquet'
    output_dir = project_root / 'model_output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        logger.info("Create output directory: {output_dir}")
    else:
        logger.info(f"{output_dir} already exists!")

    # prepare dataset
    train_dataset, eval_dataset = get_training_local(input_file_path)

    # Define custom SBERT Model
    base_model_name = "BAAI/bge-small-en"
    embedding_dim = SentenceTransformer(base_model_name).get_sentence_embedding_dimension()
    logger.info(f"embedding_dim: {embedding_dim}")
    adapter1 = LinearAdapter(embedding_dim=embedding_dim,
                             adapter_dim=embedding_dim,
                             bias=False)
    adapter2 = TwoLayerAdapter(embedding_dim=embedding_dim,
                               hidden_dim=128,
                               adapter_dim=embedding_dim,
                               bias=False,
                               add_residual=True)

    # Load training arguments
    with open(project_root / 'config/training_args.yaml') as f:
        training_args = yaml.safe_load(f)
    logger.info(f"training_args: {training_args}")

    # Define fine-tuning engine
    engine = SbertAdapterFinetuneEngine(base_model_name=base_model_name,
                                        adapter_model=adapter1,
                                        training_dataset=train_dataset,
                                        eval_dataset=eval_dataset,
                                        text_col_name="text",
                                        question_col_name="question",
                                        training_args=training_args)
    # Set up to run fine-tuning
    engine.construct_custom_sbert()
    engine.create_trainer()
    engine.train()
    engine.save_finetune_adapter(output_dir)


def setup_mlflow():
    logger.info(f"mlflow version: {mlflow.__version__}")

    tracking_uri = mlflow.get_tracking_uri()
    logger.info(f"tracking_uri: {tracking_uri}")
    if tracking_uri == f'file://{os.path.abspath("mlruns")}':
        logger.info("tracking_uri is set to the default value on the local filesystem")
        logger.info(f"tracking_uri: {tracking_uri}")
        logger.info("Configure the tracking URI to point to the MLflow server")
        tracking_uri = "http://localhost:5000"
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"tracking_uri now set to: {tracking_uri}")
    elif tracking_uri.startswith("databricks"):
        print("tracking_uri is set to the Databricks tracking server")
    else:
        print("tracking_uri is unknown")


if __name__ == '__main__':
    setup_mlflow()
    run_finetune()
