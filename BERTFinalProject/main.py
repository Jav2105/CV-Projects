
import transformers
from transformers import (set_seed, utils)
from datasets import load_dataset
import evaluate
import numpy as np
from pipeline import pipeline
from my_tokenizers import bert_tokenizer, gpt_tokenizer

set_seed(42)

# To reduce output clutter
transformers.logging.set_verbosity_error()
utils.logging.disable_progress_bar()

# CLARIFICATION ON THE NAMES USED FOR THE ARCHITECTURES:
# - NAIVE: BERT with 100-0-0 masking
# - IMPROVED: BERT with default 80-10-10 masking
# - GPT: OpenAI GPT 1

# PRETRAINING (MLM)
# We load the Wikitext data
pretraining_data = load_dataset("wikitext", "wikitext-2-raw-v1")

def pretraining_bert_tokenize_function(examples):
    return bert_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

def pretraining_gpt_tokenize_function(examples):
    return gpt_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

def pretraining_metrics(pred):
    metrics = evaluate.load("accuracy")
    predictions, labels = pred
    # Ignore -100 for GPT 1 (not doing so causes bugs)
    predictions = predictions[labels != -100]
    labels = labels[labels != -100]
    return metrics.compute(predictions=predictions, references=labels)

pipeline(task = "PRETRAINING", architecture="NAIVE", global_vars=globals())
pipeline(task = "PRETRAINING", architecture="IMPROVED", global_vars=globals())
pipeline(task = "PRETRAINING", architecture="GPT", global_vars=globals())

# SST2 EVALUATION
# Objective: classify sentences to decide whether they are positive or negative.
 
# Load data
sst2_data = load_dataset("glue", "sst2")

# Tokenization
# We need to redefine the tokenization functions for SST-2
def sst2_bert_tokenize_function(examples):
    # sentence is used instead of text
    return bert_tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

def sst2_gpt_tokenize_function(examples):
    result = gpt_tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)
    # Supervised for classification instead of self-supervised
    result["labels"] = examples["label"]
    return result

def sst2_metrics(pred):
    metrics = evaluate.load("glue", "sst2")
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return metrics.compute(predictions=predictions, references=labels)

pipeline(task = "SST2", architecture="NAIVE", global_vars=globals())
pipeline(task = "SST2", architecture="IMPROVED", global_vars=globals())
pipeline(task = "SST2", architecture="GPT", global_vars=globals())

# MRPC EVALUATION
# Objective: determine whether two sentences are semantically equivalent.

# Load data
mrpc_data = load_dataset("glue", "mrpc")

def mrpc_bert_tokenize_function(examples):
    # Sentences go in pairs
    return bert_tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", 
                          max_length=128)

def mrpc_gpt_tokenize_function(examples):
    result = gpt_tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", 
                            max_length=128)
    # Supervised for classification instead of self-supervised
    result["labels"] = examples["label"]
    return result

def mrpc_metrics(pred):
    metrics = evaluate.load("glue", "mrpc")
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return metrics.compute(predictions=predictions, references=labels)

pipeline(task = "MRPC", architecture="NAIVE", global_vars=globals())
pipeline(task= "MRPC", architecture="IMPROVED", global_vars=globals())
pipeline(task = "MRPC", architecture="GPT", global_vars=globals())

# RTE EVALUATION
# Objective: calculate whether a premise entails a hypothesis or not.
# Used instead of MNLI because of computing constraints.

rte_data = load_dataset("glue", "rte")

print(rte_data["train"].column_names)

def rte_bert_tokenize_function(examples):
    # Premise and hypothesis are used
    return bert_tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", 
                          max_length=128)

def rte_gpt_tokenize_function(examples):
    result = gpt_tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", 
                            max_length=128)
    # Supervised for classification instead of self-supervised
    result["labels"] = examples["label"]
    return result

def rte_metrics(pred):
    metrics = evaluate.load("glue", "rte")
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return metrics.compute(predictions=predictions, references=labels)

pipeline(task = "RTE", architecture="NAIVE", global_vars=globals())
pipeline(task = "RTE", architecture="IMPROVED", global_vars=globals())
pipeline(task = "RTE", architecture="GPT", global_vars=globals())