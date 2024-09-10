# Fine-Tuning Llama 2 for Question Answering and BERT for Named Entity Recognition (NER)

## Overview

This repository contains two fine-tuned models for:
1. **Question Answering** using the Llama 2 model.
2. **Named Entity Recognition (NER)** using the BERT model. 

Both models are fine-tuned on specific datasets and optimized for their respective tasks. The code provided demonstrates how to fine-tune, evaluate, and use these models for real-world applications.

---

## Models

### 1. **Llama 2 - Question Answering**
   - **Model:** Llama 2
   - **Task:** Fine-tuned for Question Answering (QA) tasks.
   - **Usage:** Given a context and a question, the model generates precise answers based on the input.

### 2. **BERT - Named Entity Recognition (NER)**
   - **Model:** BERT (bert-base-cased)
   - **Task:** Fine-tuned for NER, which involves identifying and classifying named entities (like people, organizations, locations) in text.
   - **Dataset:** CoNLL-2003

---

## Setup and Requirements

To set up the environment and run the code for both tasks, ensure you have the following Python packages installed:

```bash
pip install transformers datasets tokenizers seqeval
```

---

## BERT - Named Entity Recognition (NER)

### Dataset
The **CoNLL-2003** dataset is used for fine-tuning BERT for NER. The dataset contains entities such as names of people, organizations, locations, and miscellaneous items.

### Model Fine-Tuning
The `BertTokenizerFast` is used for tokenization, and the model is trained using a `Trainer` API from HuggingFace's `transformers` library. The training process includes token alignment with entity tags and the use of a data collator for padding.

### Example Code for Fine-Tuning
```python
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = AutoModelForTokenClassification.from_pretrained('bert-base-cased', num_labels=9)

# Tokenizing and aligning labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    ...
    return tokenized_inputs

# Fine-tuning the model
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
```

### Model Inference Example
Once trained, the BERT model can be used for NER tasks as follows:
```python
from transformers import pipeline

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Bill Gates is the founder of Microsoft."
ner_results = nlp(example)
print(ner_results)
```

---

## Llama 2 - Question Answering

### Dataset
Fine-tuning of the Llama 2 model for QA tasks can be done using any QA dataset (e.g., SQuAD). The model learns to generate context-aware answers for user-provided questions.

### Example Code for Fine-Tuning Llama 2
The Llama 2 model can be loaded and fine-tuned similarly using the Hugging Face `transformers` library.

```python
from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments

model = AutoModelForQuestionAnswering.from_pretrained("llama-2")
# Followed by setting up the tokenizer, data, and training loop...
```

### Model Inference Example
Once fine-tuned, the Llama 2 model can be used as follows for question answering:
```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="llama-2", tokenizer="llama-2")
qa_results = qa_pipeline(question="Who is the founder of Microsoft?", context="Microsoft was founded by Bill Gates and Paul Allen.")
print(qa_results)
```

---

## Saving and Loading Fine-Tuned Models

After fine-tuning, the models and tokenizers can be saved for future use:
```python
model.save_pretrained('ner_model')
tokenizer.save_pretrained('tokenizer')
```

You can then load the model and tokenizer back as needed for inference:
```python
from transformers import AutoModelForTokenClassification, BertTokenizerFast

model = AutoModelForTokenClassification.from_pretrained('ner_model')
tokenizer = BertTokenizerFast.from_pretrained('tokenizer')
```

---

## Evaluation and Metrics

The models are evaluated using standard metrics like Precision, Recall, F1-Score, and Accuracy.

For the NER task, the **SeqEval** library is used for evaluation:
```python
from datasets import load_metric
metric = load_metric('seqeval')

# Example evaluation on predictions
results = metric.compute(predictions=pred_labels, references=true_labels)
print(results)
```

---

## Conclusion

This repository demonstrates how to fine-tune state-of-the-art models like BERT for Named Entity Recognition and Llama 2 for Question Answering tasks. With the provided code, you can reproduce the results or adapt the models for your own dataset and tasks.
