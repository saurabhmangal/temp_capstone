import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class LLMGeneratedDataset(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, model, vocab_words, block_size, num_samples=10000, seed=42):
        self.tokenizer = tokenizer
        # Ensure the tokenizer has a padding token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.model = model
        self.vocab_words = vocab_words
        self.block_size = block_size
        self.num_samples = num_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.num_samples):
            # Randomly select a word from the vocabulary
            word = self.rng.choice(self.vocab_words)
            # Tokenize the word and generate data
            inputs = self.tokenizer(word, return_tensors="pt", return_attention_mask=False)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=self.block_size, pad_token_id=self.tokenizer.pad_token_id)
            # Decode and then re-tokenize to ensure consistency with block_size
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            tokenized_text = self.tokenizer(text, max_length=self.block_size, padding="max_length", truncation=True, return_tensors="pt")
            yield tokenized_text.input_ids.squeeze(0)

# Initialize the Phi-2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# Extract vocabulary words
vocab_words = list(tokenizer.get_vocab().keys())

# Define block_size and number of samples
block_size = 128  # Example block size, adjust as needed
num_samples = 10000  # Define how many samples you want to generate

# Create the LLMGeneratedDataset instance
llm_dataset = LLMGeneratedDataset(tokenizer, model, vocab_words, block_size, num_samples)

