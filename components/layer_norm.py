import torch.nn as nn

class LayerNorm(nn.Module):

    """
    This class is used to create a layer normalization layer for the input tensor. It takes in the embedding dimension as input and creates a layer normalization layer using nn.LayerNorm. The forward method takes in the input tensor and returns the corresponding normalized tensor.
    """

    def __init__(self,embedding_dim):
        super(LayerNorm,self).__init__()
        self.embedding_dim = embedding_dim
        # Normalization values
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(self,x):
        out = self.layer_norm(x)
        return out