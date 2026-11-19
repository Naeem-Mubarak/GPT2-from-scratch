from torch.optim.lr_scheduler import  LinearLR, CosineAnnealingLR, SequentialLR
from config import MINI_GPT2_CONFIG
from dataset import train_data
import torch.nn as nn
from GPT2 import GPT2
import torch


vocab_size = MINI_GPT2_CONFIG["vocab_size"]
embedding_dim = MINI_GPT2_CONFIG["embedding_dim"]
context_size = MINI_GPT2_CONFIG["context_size"]
n_head = MINI_GPT2_CONFIG["n_head"]
dropout_ratio = MINI_GPT2_CONFIG["dropout_ratio"]


# cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = GPT2(vocab_size,embedding_dim,context_size,n_head,dropout_ratio)
model = model.to(device)


epochs = 20
best_val_loss = float('inf')

# LR warm up
initial_lr = 3e-6
peak_lr = 1e-3

# decay
total_steps = epochs * len(train_data)
warmup_steps = int(0.05 * total_steps)
decay_steps = total_steps - warmup_steps
min_lr = 1e-5


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=peak_lr,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1
)

# Loss function
loss_fn = nn.CrossEntropyLoss()



# LR Schduler (To stabalize the training and avoid overshooting)
schduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(
            optimizer,
            start_factor=0.003,
            end_factor=1.0,
            total_iters=warmup_steps
            ),
        CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=min_lr)
    ],
    milestones=[warmup_steps]
)