from torch.utils.data import Dataset
import torch
import math

class DataPreparation(Dataset):

  # This class is used to prepare the data for training and testing. It takes in a tokenizer, the dataset, and the context size as input. It tokenizes the data, creates input-output pairs (X and y) by shifting the tokens, and provides methods to get the length of the dataset and retrieve individual samples.

  def __init__(self,tokenizer,data,context_size):

    self.tokenizer = tokenizer # tokenizer object
    self.data = data # dataset
    self.context_size = context_size # context window
    self.X , self.y = self._train_testing_data()

  def _train_testing_data(self):
    X = []
    y = []

    # Tokenization (getting token ids)
    tokens = self.tokenizer.encode(self.data)
    if tokens[-1] != self.tokenizer.eot_token:
      tokens.append(self.tokenizer.eot_token)

    # discarding extra tokens
    samples = math.floor((len(tokens)-1)/self.context_size)
    samples = samples * self.context_size + 1
    final_tokens = tokens[:samples]

    # Getting X and y out of actaul data by shifiting one position
    for i in range(0,len(final_tokens)-1,self.context_size):
      X.append(final_tokens[i:i+self.context_size])
      y.append(final_tokens[i+1:i+self.context_size+1])

    return torch.tensor(X, dtype=torch.long),torch.tensor(y, dtype=torch.long)

  def __len__(self):
    return len(self.X)

  def __getitem__(self,idx):
    # providing x,y one by one to data loader to class to make batches
    return self.X[idx],self.y[idx]