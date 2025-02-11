from transformers import BertTokenizer

# Download the pre-trained Nepali BERT tokenizer (replace with the correct Nepali BERT model name)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")  # Use multilingual BERT for Nepali

# Sample text in Nepali
text = "म काठमाडौँ जाँदैछु।"

# Tokenize the text
tokens = tokenizer.encode(text, add_special_tokens=True)  # Adds [CLS] and [SEP] tokens
print(f"Encoded tokens: {tokens}")

# Decode the tokens back to text
decoded_text = tokenizer.decode(tokens)
print(f"Decoded text: {decoded_text}")

# Another example text
tokens = tokenizer.encode("काठमाडौँको मौसम सुन्दर छ।", add_special_tokens=True)
print(f"Encoded tokens: {tokens}")
decoded_text = tokenizer.decode(tokens)
print(f"Decoded text: {decoded_text}")

# Save the tokenizer to the current directory
tokenizer.save_pretrained("./tokenizer")

vocab_size = len(tokenizer)
print("Vocabulary size:", vocab_size)