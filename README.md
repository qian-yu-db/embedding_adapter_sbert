# Embedding Adapter Fine-tuning with Sentence Transformer

A framework for fine-tuning Sentence Transformer embedding models using custom adapters. This project enables efficient adaptation of pre-trained embedding models for domain-specific tasks through lightweight adapter layers.

## Features

- Custom adapter implementations (Linear, Two-layer)
- Fine-tuning engine for Sentence Transformers
- MLflow integration for experiment tracking
- Support for both local and Databricks environments
- Comprehensive evaluation pipeline

## Project Structure

- **`sbert_adapter/`** - Core Python package containing adapter implementations, fine-tuning engine, and main execution logic
- **`notebooks/`** - Jupyter notebooks demonstrating the complete workflow from model deployment to evaluation
- **`config/`** - Configuration files for training arguments and hyperparameters
- **`tests/`** - Unit tests for adapter and fine-tuning engine components
