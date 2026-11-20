import torch.nn as nn
import torch

class TokenEmbedding(nn.Module):

  """
  This class is used to create an embedding layer for the input tokens. It takes in the vocabulary size and the embedding dimension as input and creates an embedding layer using nn.Embedding. The forward method takes in the input tokens and returns the corresponding embeddings.
  """

  def __init__(self,vocab_size,embedding_dim):
    super(TokenEmbedding,self).__init__()
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    # embeeding layer
    self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)

  def forward(self,tokens):
    out = self.embedding(tokens)
    return out



class PositionalEmbedding(nn.Module):

    """
    This class is used to create a positional embedding layer for the input tokens. It takes in the context size and the embedding dimension as input and creates an embedding layer using nn.Embedding. The forward method takes in the input tokens and returns the corresponding positional embeddings based on the index of each token.
    """
    def __init__(self,context_size,embedding_dim):

        super(PositionalEmbedding,self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.pos_embedding = nn.Embedding(self.context_size,self.embedding_dim)

    def forward(self,tokens):

        batch_size, seq_len = tokens.shape
        # getting based on the index value of each token (learnable embeddings)
        positions = torch.arange(0,seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        out = self.pos_embedding(positions)
        return out