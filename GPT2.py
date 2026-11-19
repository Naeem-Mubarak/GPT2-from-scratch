import torch.nn as nn
from components.layer_norm import LayerNorm
from components.embedding import PositionalEmbedding
from components.embedding import Embedding
from Transformer_block import TransformerBlock



class GPT2(nn.Module):

  """
  This class is used to create a GPT2 model for the input tensor. It takes in the vocabulary size, embedding dimension, context size, number of heads, and dropout ratio as input and creates a GPT2 model using transformer blocks, positional embedding, and token embedding. The forward method takes in the input tensor and returns the corresponding output tensor.
  """

  def __init__(self,vocab_size,embedding_dim,context_size,n_head,dropout_ratio):

    super(GPT2,self).__init__()
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.context_size = context_size
    self.n_head = n_head
    self.dropout_ratio = dropout_ratio
    self.dropout = nn.Dropout(dropout_ratio)

    self.final_layer_norm = LayerNorm(self.embedding_dim)
    self.pos_embedding = PositionalEmbedding(self.context_size, self.embedding_dim)
    self.token_embedding = Embedding(self.vocab_size, self.embedding_dim)

    self.LMHead = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)

    # weight tying (Sharing same weights for token embedding & LM head)
    self.LMHead.weight = self.token_embedding.embedding.weight

    # using 12 Transformer Blocks
    self.transformer_blocks = nn.ModuleList([
        TransformerBlock(self.embedding_dim, self.n_head, self.dropout_ratio)
        for _ in range(12)
    ])

  def forward(self,x):
    token_embedding = self.token_embedding(x)
    pos_embedding = self.pos_embedding(x)
    x = token_embedding + pos_embedding
    x = self.dropout(x)

    for block in self.transformer_blocks:
      x = block(x)

    output_layer_norm = self.final_layer_norm(x)

    out = self.LMHead(output_layer_norm)

    return out