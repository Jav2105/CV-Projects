
from transformers import (BertConfig, OpenAIGPTConfig)
from my_tokenizers import bert_tokenizer, gpt_tokenizer

# CONFIGURATION
bert_config = BertConfig(
    # Original paper configuration: BERT base.
    vocab_size=bert_tokenizer.vocab_size, # Default vocab size
    hidden_size=256, # Size of vector embeddings. Original paper configuration: 768.
    num_hidden_layers=4, # Original paper configuration: 12
    num_attention_heads=4, # Original paper configuration: 12
    intermediate_size=1024, # 4*256. Original paper configuration: 3072 (4*768)
    max_position_embeddings=128
)

gpt_config = OpenAIGPTConfig(
    vocab_size = gpt_tokenizer.vocab_size, # Default vocab size
    # Same values as the BERT model
    n_embd = 256, # Dimension of token representations
    n_layer = 4,
    n_head = 4,     
    n_inner = 1024,
    n_positions = 128,
    pad_token_id=gpt_tokenizer.pad_token_id
)