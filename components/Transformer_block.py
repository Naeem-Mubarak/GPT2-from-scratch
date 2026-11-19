import torch.nn as nn
from components.MHA import MultiHeadAttention
from components.ffn import Feed_Forward_Layer
from components.layer_norm import LayerNorm

class TransformerBlock(nn.Module):

  """
  This class is used to create a transformer block for the input tensor. It takes in the embedding dimension, number of heads, and dropout ratio as input and creates a transformer block using multi-head attention and feed forward layers. The forward method takes in the input tensor and returns the corresponding output tensor"""

  def __init__(self, embedding_dim, n_head, dropout_ratio):
    super(TransformerBlock,self).__init__()

    self.embedding_dim = embedding_dim
    self.n_head = n_head
    self.dropout_ratio = dropout_ratio

    self.ln_1 = LayerNorm(self.embedding_dim)
    self.attention = MultiHeadAttention(self.embedding_dim,self.n_head,self.dropout_ratio)
    self.ln_2 = LayerNorm(self.embedding_dim)
    self.mlp = Feed_Forward_Layer(self.embedding_dim)
    self.dropout1 = nn.Dropout(dropout_ratio)
    self.dropout2 = nn.Dropout(dropout_ratio)

  def forward(self,x):
    layer1 = self.ln_1(x)
    attention_output = self.attention(layer1)
    resid_dropout1 = self.dropout1(attention_output)
    resid1 = x + resid_dropout1

    # resid connection is also used as input to the feed forward layer
    layer2 = self.ln_2(resid1)
    mlp_output = self.mlp(layer2)
    resid_dropout2 = self.dropout2(mlp_output)
    resid2 = resid1 + resid_dropout2

    return resid2