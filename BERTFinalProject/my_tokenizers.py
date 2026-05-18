
from transformers import AutoTokenizer

# Tokenizers for both BERT and GPT
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt_tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
# BERT does not need an assigned pad_token
gpt_tokenizer.pad_token = gpt_tokenizer.unk_token