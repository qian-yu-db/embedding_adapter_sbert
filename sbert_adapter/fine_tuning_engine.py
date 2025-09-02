import logging

import mlflow
from datasets import Dataset
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from .adapters import *

logging.basicConfig()
logger = logging.getLogger("fine_tuning_engine")
logger.setLevel(logging.INFO)


class SbertAdapterFinetuneEngine():
    """Sentence Transformers Adapter Finetune Engine."""

    def __init__(self,
                 base_model_name,
                 adapter_model,
                 training_dataset,
                 eval_dataset,
                 text_col_name,
                 question_col_name,
                 training_args,
                 ) -> None:

        # Create sentence transformer triplet dataset
        column_remap = {text_col_name: "anchor", question_col_name: "positive"}

        self.ft_train_dataset = Dataset.from_pandas(
            training_dataset.rename(columns=column_remap),
            preserve_index=False
        )
        self.ft_eval_dataset = Dataset.from_pandas(
            eval_dataset.rename(columns=column_remap),
            preserve_index=False
        )
        logger.info(f"ft_train_dataset: {self.ft_train_dataset}")
        logger.info(f"ft_eval_dataset: {self.ft_eval_dataset}")

        # Load base model
        self.base_model = SentenceTransformer(base_model_name, trust_remote_code=True)
        self.embedding_dim = self.base_model.get_sentence_embedding_dimension()

        # Define adapter model
        if adapter_model is not None:
            self.adapter_model = adapter_model
        else:
            self.adapter_model = LinearAdapter(embedding_dim=self.embedding_dim,
                                               adapter_dim=self.embedding_dim,
                                               bias=False)

        # Define training arguments
        self.training_args = SentenceTransformerTrainingArguments(**training_args)

        # Define a model placeholder
        self.model = None

    def construct_custom_sbert(self):
        transformer_module = self.base_model[0]
        pooling_module = self.base_model[1]
        normalize_module = self.base_model[2]

        # Freeze transformer module to ensure based embedding model is frozen
        transformer_module.eval()
        for param in transformer_module.parameters():
            param.requires_grad = False

        # Construct custom SBERT model
        custom_sbert = SentenceTransformer(modules=[transformer_module,
                                                    self.adapter_model,
                                                    pooling_module,
                                                    normalize_module])

        # Test custom SBERT model
        try:
            embedding_example = custom_sbert.encode(["Hello, World!"])
            logger.info(f"embedding_example: {len(embedding_example[0])}")
        except Exception as e:
            logger.error(f"Custom SBERT model failed to encode: {e}")

        return custom_sbert

    def create_trainer(self):
        self.model = self.construct_custom_sbert()
        loss = CachedMultipleNegativesRankingLoss(self.model, mini_batch_size=8)
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=self.training_args,
            loss=loss,
            train_dataset=self.ft_train_dataset,
            eval_dataset=self.ft_eval_dataset
        )
        return trainer

    def train(self, run_name="sbert_finetune_run"):
        trainer = self.create_trainer()
        with mlflow.start_run(run_name=run_name) as run:
            trainer.train()

    def save_finetune_adapter(self, save_dir):
        # the adapter model is between transformer and pooling in the module sequence
        finetuned_adapter = self.model[1]
        finetuned_adapter.save(save_dir)

    def get_finetune_adapter(self):
        return self.model[1]
