from config import model, MINI_GPT2_CONFIG
from training.training_recipe import epochs, optimizer, schduler, loss_fn, device, train_data, val_data
from training.saving_checkpoints import save_checkpoint, latest_checkpoint
import torch


batch_size = MINI_GPT2_CONFIG["batch_size"]
context_size = MINI_GPT2_CONFIG["context_size"]
vocab_size = MINI_GPT2_CONFIG["vocab_size"]


training_loss = []
validation_loss = []
for epoch in range(epochs):

  # Training Phase
  model.train()
  batch_loss = 0.0

  for batch_x, batch_y in train_data:

    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    optimizer.zero_grad()

    # forward pass
    output = model(batch_x)

    output = output.reshape(batch_size * context_size,vocab_size)
    batch_y = batch_y.reshape(batch_size * context_size)


    # Calculate loss
    loss = loss_fn(output,batch_y)

    # backward pass
    loss.backward()

    # gradient clipping
    total_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
    )

    optimizer.step()
    schduler.step()

    batch_loss = batch_loss + loss.item()
  training_loss.append(batch_loss/len(train_data))

  # Validation Phase
  val_loss = 0.0
  with torch.inference_mode():
    model.eval()

    for batch_x, batch_y in val_data:
      batch_x = batch_x.to(device)
      batch_y = batch_y.to(device)

      # forward pass
      output = model(batch_x)
      output = output.reshape(batch_size * context_size,vocab_size)
      batch_y = batch_y.reshape(batch_size * context_size)

      # loss
      loss = loss_fn(output,batch_y)

      val_loss = val_loss + loss.item()
    avg_val_loss = val_loss/len(val_data)
    validation_loss.append(val_loss/len(val_data))


    if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      save_checkpoint(model, optimizer, schduler, epoch, best_val_loss)
    latest_checkpoint(model, optimizer, schduler, epoch, best_val_loss)

  print(f"Epoch: {epoch+1}, Training Loss: {batch_loss/len(train_data)}, Validation Loss: {val_loss/len(val_data)}")