from torch.utils.data import Dataset, DataLoader
from components.tokenizer import Tokenizer
from config import MINI_GPT2_CONFIG, tokenizer
import torch
import math

batch_size = MINI_GPT2_CONFIG["batch_size"]
context_size = MINI_GPT2_CONFIG["context_size"]


# The dataset is not ideal for training i extracted it from one of the best Books on AI (14 books)
with open("../data/dataset.txt", "r", encoding="utf-8") as f:
    data = f.read()


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


split_idx = int(len(data) * 0.9)
train_data = data[:split_idx]
val_data = data[split_idx:]


# Wrap the raw tiktoken tokenizer with your custom Tokenizer wrapper to allow special tokens
custom_tokenizer = Tokenizer(tokenizer)

train_dataset = DataPreparation(custom_tokenizer, train_data, context_size)
val_dataset = DataPreparation(custom_tokenizer, val_data, context_size)


train_data = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    drop_last=True
  )

val_data = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    drop_last=True
  )