import torch.nn as nn
import math
import torch


class MultiHeadAttention(nn.Module):

    """
    This class is used to create a multi-head attention layer for the input tensor. It takes in the embedding dimension, number of heads, and dropout ratio as input and creates a multi-head attention layer. The forward method takes in the input tensor and returns the corresponding attention output.
    """

    def __init__(self,e_dim,n_head,dropout_ratio):

        super(MultiHeadAttention,self).__init__()

        self.e_dim = e_dim
        self.n_head = n_head
        self.dropout_ratio = dropout_ratio
        self.dropout = nn.Dropout(self.dropout_ratio)

        # initializing Query, Key and Value matrix
        self.q_linear = nn.Linear(self.e_dim,self.e_dim)
        self.k_linear = nn.Linear(self.e_dim,self.e_dim)
        self.v_linear = nn.Linear(self.e_dim,self.e_dim)

        self.weight_matrix = nn.Linear(self.e_dim,self.e_dim)

    def forward(self,x):

        if self.e_dim % self.n_head != 0:
            raise ValueError("Embedding dimension should be divisible by number of heads")

        batch_size, seq_len, e_dim = x.shape
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Multihead attention (MHA) making 6 attention blocks and separate Q,K,V matrix
        Q = q.view(
            batch_size,
            seq_len,
            self.n_head,
            e_dim//self.n_head
            ).transpose(1,2)

        K = k.view(
            batch_size,
            seq_len,
            self.n_head,
            e_dim//self.n_head
            ).transpose(1,2)
        V = v.view(
            batch_size,
            seq_len,
            self.n_head,
            e_dim//self.n_head
            ).transpose(1,2)

        # Attention score calculation
        O = Q@K.transpose(-2,-1)

        # scaling factor is independent for every head
        scaling_factor = e_dim/self.n_head
        scaling = math.sqrt(scaling_factor)

        O = O/scaling

        # Casual Masking to prevent model from cheating (paying attention to future tokens)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device),diagonal=1).bool()
        O = O.masked_fill(mask,float('-inf'))

        # Softmax
        A = torch.softmax(O,dim=-1)
        A = self.dropout(A)

        # Attention output
        attention_output = torch.matmul(A,V)

        # reshaping
        attention_output = attention_output.transpose(1,2).contiguous()

        # concatinating the output of all heads
        attention_output = attention_output.view(batch_size,seq_len,e_dim)

        # W_O weight matrix
        out = self.weight_matrix(attention_output)
        return out