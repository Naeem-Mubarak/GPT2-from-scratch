import torch.nn as nn


class Feed_Forward_Layer(nn.Module):

    """
    This class is used to create a feed forward layer for the input tensor. It takes in the embedding dimension as input and creates a feed forward layer using nn.Linear and nn.GELU. The forward method takes in the input tensor and returns the corresponding output tensor.
    """

    def __init__(self,embedding_dim):
        super(Feed_Forward_Layer,self).__init__()
        self.embedding_dim = embedding_dim
        self.Linear = nn.Sequential(
            # feature mapping
            # original architechture use 4 times the embedding dimension but we are using 2 times the embedding dimension to reduce the number of parameters
            nn.Linear(self.embedding_dim,2*self.embedding_dim),
            nn.GELU(),
            nn.Linear(2*self.embedding_dim,self.embedding_dim)
        )

    def forward(self,x):
        out = self.Linear(x)
        return out