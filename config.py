import tiktoken
tokenizer = tiktoken.encoding_for_model("gpt-2")


ORIGINAL_GPT2_CONFIG = {
    "context_size" : 512, # maximum context model can process at a time
    "batch_size" : 0, # Don't know about that
    "embedding_dim" : 768, # dimension of vector of each token
    "vocab_size" : tokenizer.n_vocab, # using the vocab size of gpt tokenizer
    "n_head" : 12, # number of attention head
    "dropout_ratio" : 0.1 # dropout
}


MINI_GPT2_CONFIG = {
    "context_size" : 28, 
    "batch_size" : 8,
    "embedding_dim" : 192,
    "vocab_size" : tokenizer.n_vocab,
    "n_head" : 6,
    "dropout_ratio" : 0.1
}