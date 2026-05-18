
from transformers import (BertForMaskedLM, OpenAIGPTLMHeadModel, DataCollatorForLanguageModeling, TrainingArguments, 
                          Trainer, BertForSequenceClassification, OpenAIGPTForSequenceClassification)
import os
from my_tokenizers import bert_tokenizer, gpt_tokenizer
from configs import bert_config, gpt_config

# If we want to retrain the models or not
RETRAIN = False

# Number of labels for each task
# Flexible for addition of new tasks
NUMBER_LABELS = {"SST2": 2, "MRPC": 2, "RTE": 2}

# CLARIFICATION ON THE NAMES USED FOR THE ARCHITECTURES:
# - NAIVE: BERT with 100-0-0 masking
# - IMPROVED: BERT with default 80-10-10 masking
# - GPT: OpenAI GPT 1

# We define a pipeline to make the code more compact, legible and reusable
def pipeline(task: str, architecture: str, global_vars: dict):
    '''Main backbone of the code, which trains and evaluates the model.'''

    if task not in NUMBER_LABELS.keys() and task != "PRETRAINING":
        raise ValueError("Not a valid task")

    if architecture not in ("NAIVE", "IMPROVED", "GPT"):
        raise ValueError("Not a valid architecture") 
    
    pretraining = task == "PRETRAINING"
    # Where we save the results
    results_path = f"{task}_{architecture}_results".lower()
    # Where we save the model for future retraining
    saved_path = f"{task}_{architecture}_saved".lower()

    # We create our model, or we load it from storage
    if os.path.exists(saved_path) and not RETRAIN:
        if pretraining:
            if architecture == "GPT":
                model = OpenAIGPTLMHeadModel.from_pretrained(saved_path)
            else:
                model = BertForMaskedLM.from_pretrained(saved_path)
        else:
            if architecture == "GPT":
                model = OpenAIGPTForSequenceClassification.from_pretrained(saved_path)
            else:
                model = BertForSequenceClassification.from_pretrained(saved_path)
    elif not pretraining:
        if architecture == "GPT":
            # We load the model obtained from the pretraining
            model = OpenAIGPTForSequenceClassification.from_pretrained(f"pretraining_{architecture.lower()}_saved", 
                                                                       num_labels = NUMBER_LABELS[task])
        else:
            # We load the model obtained from the pretraining
            model = BertForSequenceClassification.from_pretrained(f"pretraining_{architecture.lower()}_saved", 
                                                                  num_labels = NUMBER_LABELS[task])
    else:
        if architecture == "GPT":
            model = OpenAIGPTLMHeadModel(gpt_config)
        else:
            model = BertForMaskedLM(bert_config)

    # We define the collator to use in pretraining
    if pretraining:
        if architecture == "GPT":
            # MLM is only used for BERT, it is incompatible with GPT
            collator = DataCollatorForLanguageModeling(gpt_tokenizer, mlm=False)
        elif architecture == "NAIVE":
            # 100-0-0 masking configuration
            collator = DataCollatorForLanguageModeling(bert_tokenizer, mlm=True, mask_replace_prob=1, random_replace_prob=0)
        elif architecture == "IMPROVED":
            # Default 80-10-10 masking configuration
            collator = DataCollatorForLanguageModeling(bert_tokenizer, mlm=True)
    else:
        collator = None

    # Necessary to avoid bugs in pretraining, not necessary for fine-tuning
    def preprocess_logic_for_metrics(logits, labels):
            return logits.argmax(dim=-1)

    # We extract the global variables for the tokenizer function, the data and the metrics
    gpt_bert = "gpt" if architecture == "GPT" else "bert"
    tokenize_function = global_vars[f"{task.lower()}_{gpt_bert}_tokenize_function"]
    data = global_vars[f"{task.lower()}_data"]
    metrics = global_vars[f"{task.lower()}_metrics"]

    # We tokenize the data
    tokenized_data = data.map(tokenize_function, batched=True)
    # We leave the default configuration for the training arguments, disabling tqdm to reduce clutter and
    # setting eval_accumulation_steps to avoid problems with CPU/GPU coordination
    training_arguments = TrainingArguments(results_path, disable_tqdm=True, eval_accumulation_steps=10)

    if pretraining:
        trainer = Trainer(model, training_arguments, collator, train_dataset=tokenized_data["train"], 
                      eval_dataset=tokenized_data["validation"], compute_metrics=metrics, 
                      preprocess_logits_for_metrics=preprocess_logic_for_metrics)
    else:
        # MNLI has a different name for its validation dataset
        trainer = Trainer(model, training_arguments, collator, train_dataset=tokenized_data["train"], 
                      eval_dataset=tokenized_data["validation_matched" if task=="MNLI" else "validation"], compute_metrics=metrics)
        
    # Training can be disabled for efficiency
    if not os.path.exists(saved_path) or RETRAIN:
        trainer.train()
        trainer.save_model(saved_path)

    print(f"{task} {architecture} model:")
    trainer.evaluate()