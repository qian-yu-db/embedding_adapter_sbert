import pytest
from src.sbert_adapter.fine_tuning_engine import SbertAdapterFinetuneEngine
from sentence_transformers import SentenceTransformer
from src.sbert_adapter.adapters import LinearAdapter
import pandas as pd

@pytest.fixture
def training_data():
    data = {
        'id': ['1', '2'],
        'text': ['This is a test sentence.', 'Another test sentence.'],
        'question': ['What is this?', 'What is that?']
    }
    return pd.DataFrame(data)

@pytest.fixture
def eval_data():
    data = {
        'id': ['3'],
        'text': ['Yet another test sentence.'],
        'question': ['What is this?']
    }
    return pd.DataFrame(data)

@pytest.fixture
def embedding_dim():
    base_model_name = "BAAI/bge-small-en"
    return SentenceTransformer(base_model_name).get_sentence_embedding_dimension()

@pytest.fixture
def linear_adapter(embedding_dim):
    return LinearAdapter(embedding_dim=embedding_dim, adapter_dim=embedding_dim, bias=False)

@pytest.fixture
def training_args():
    return {
        'num_train_epochs': 1,
        'learning_rate': 1e-4,
        'output_dir': "model_outputs"
    }

def test_sbert_adapter_finetune_engine_initialization(embedding_dim, linear_adapter, training_data, eval_data, training_args):
    engine = SbertAdapterFinetuneEngine(
        base_model_name="BAAI/bge-small-en",
        adapter_model=linear_adapter,
        training_dataset=training_data,
        eval_dataset=eval_data,
        text_col_name="text",
        question_col_name="question",
        training_args=training_args
    )
    assert engine is not None
    assert engine.base_model_name == "BAAI/bge-small-en"
    assert engine.adapter_model == linear_adapter
    assert engine.training_args.num_train_epochs == training_args['num_train_epochs']
    assert engine.training_args.learning_rate == training_args['learning_rate']
    assert engine.training_args.output_dir == training_args['output_dir']
