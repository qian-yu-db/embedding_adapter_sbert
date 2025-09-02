from sentence_transformers import SentenceTransformer, models


def rebuild_sbert_from_finetune_adapter(base_model_name,
                                        adapter,
                                        ft_adapter_path):
    """rebuild a SBERT model from fine-tuned adapter model binary."""
    embedding_dim = SentenceTransformer(base_model_name).get_sentence_embedding_dimension()
    adapter_module = adapter.load(ft_adapter_path)
    transformer_module = models.Transformer(base_model_name)
    pooling_module = models.Pooling(embedding_dim, pooling_mode="mean")
    normalize_module = models.Normalize()

    # Construct a new SBERT model
    ft_sbert_w_adapter = SentenceTransformer(modules=[transformer_module,
                                                      adapter_module,
                                                      pooling_module,
                                                      normalize_module])
    return ft_sbert_w_adapter
