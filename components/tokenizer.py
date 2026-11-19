class Tokenizer:

  # Tokenizing the dataset (Converting words into numbers and vice versa)

  def __init__(self,tokenizer):
    self.tokenizer = tokenizer

  @property
  def eot_token(self):
    return self.tokenizer.eot_token

  def encode(self,text):
    # encoding text to ids
    tokens = self.tokenizer.encode(
        text,
        allowed_special={"<|endoftext|>"}
    )
    return tokens

  def decode(self,tokens):
    # decoding ids to text
    text = self.tokenizer.decode(tokens)
    return text