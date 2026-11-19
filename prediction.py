import torch


def softmax_with_temp(input_vec, temprature):
  input_vec_temp = input_vec/temprature
  input_vec_exp = torch.exp(input_vec_temp)
  denom = torch.sum(input_vec_exp)
  return input_vec_exp/denom



def text_generation(model, tokenizer, query, max_length = 100, temprature = 0.5):
  # tokenizing the text first
  input_Tokens = tokenizer.encode(query)
  # converting them to tensor (making a batch: [1, seq_len])
  input_tensor = torch.tensor(input_Tokens).unsqueeze(0).to(device)

  for _ in range(max_length):
    # Prevent gradient calculations to save memory and speed up inference
    with torch.no_grad():
      prediction = model(input_tensor)
    prediction = prediction.squeeze(0)
    prediction = prediction[-1:]
    prediction = prediction.squeeze(0)

    # getting logits (probability of each token)
    logits = softmax_with_temp(prediction, temprature)

    # getting the index of the token with the highest probability
    idx = torch.argmax(logits, dim=-1).tolist()
    # Create new token tensor directly on the target device to avoid mismatch
    next_token_tensor = torch.tensor([idx], device=device)

    # concatenating current input and predicted token (autoregression)
    next_token_tensor = next_token_tensor.unsqueeze(0)
    input_tensor = torch.cat((input_tensor,next_token_tensor),dim=1)

  # decoding the index into words
  output = input_tensor.squeeze(0).tolist()
  output = tokenizer.decode(output)
  
  return output