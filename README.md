# Credit Risk Prediction using QLoRA Fine-tuned LLaMA

This repository contains code for a credit risk prediction model that uses QLoRA (Quantized Low-Rank Adaptation) to fine-tune the LLaMA 3.2-1B language model for binary classification of loan applications.

## Overview

The project implements a credit risk assessment system that classifies loan applications as either 'good' (likely to be repaid) or 'bad' (high risk of default) using natural language processing techniques. The model processes loan application data in a text format and leverages the power of large language models through efficient fine-tuning.

## Technical Approach

### Model Architecture
- Base Model: Meta's LLaMA 3.2-1B
- Fine-tuning Method: QLoRA (Quantized Low-Rank Adaptation)
- Task Type: Binary Sequence Classification

### Key Components

1. **Data Preprocessing**
   - Conversion of structured loan data into natural language format
   - Label encoding ('good' vs 'bad' loans)
   - Text tokenization with dynamic padding
   - Train/validation/test split (70/15/15)

2. **QLoRA Implementation**
   - 4-bit quantization for memory efficiency
   - Low-rank adaptation of key transformer components
   - Target modules: query, key, value, and output projections
   - Rank (r): 16
   - Alpha: 32
   - Dropout: 0.05

3. **Training Configuration**
   - Learning rate: 2e-4
   - Batch size: 8 (per device)
   - Gradient accumulation steps: 2
   - Weight decay: 0.01
   - BF16 mixed precision training
   - Evaluation strategy: Per epoch
   - Optimizer: Paged AdamW (32-bit)

### Performance Metrics
- F1 Score
- AUC-ROC Score

## Features

- Efficient memory usage through 4-bit quantization
- Integration with Hugging Face's Transformers library
- Custom metrics computation
- Structured training pipeline
- Model checkpointing and evaluation

## Dependencies

```
transformers
torch
datasets
bitsandbytes
accelerate
peft
scikit-learn
pandas
numpy
```

## Usage

1. Data Preparation:
```python
# Prepare your loan data in DataFrame format
df = pd.read_csv('your_loan_data.csv')
```

2. Model Training:
```python
trainer, model, tokenizer, test_results = train_model(df)
```

3. Evaluation:
```python
# Print final metrics
print(f"F1 Score: {test_results['eval_f1_score']:.4f}")
print(f"AUC Score: {test_results['eval_auc_score']:.4f}")
```

## Implementation Details

### Model Training Pipeline

The training pipeline consists of several key components:

1. **Data Processing**
   - Text generation from structured data
   - Label encoding and dataset preparation
   - Dynamic tokenization with padding

2. **Model Configuration**
   - Quantization setup for memory efficiency
   - LoRA adapter configuration
   - Training arguments optimization

3. **Training Loop**
   - Batch processing with gradient accumulation
   - Regular evaluation and metric computation
   - Model checkpointing and saving

### Metrics Computation

The model's performance is evaluated using:
- F1 Score: Measures balance between precision and recall
- AUC-ROC Score: Evaluates classification performance across different thresholds

## Results

The model achieves:
- F1 Score: 0.8885
- AUC Score: 0.7167

These metrics indicate strong performance in identifying both good and bad loans, with balanced precision and recall.

## Future Improvements

- Implement cross-validation for more robust evaluation
- Experiment with different prompt engineering techniques
- Add support for multi-class classification
- Integrate model explainability tools
- Optimize hyperparameters through systematic search
