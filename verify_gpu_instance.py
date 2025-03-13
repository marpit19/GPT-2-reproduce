import torch
import numpy as np
import os
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {n_gpus}")
    
    for i in range(n_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("No CUDA available. Running on CPU only.")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load GPT-2 model and tokenizer
print("Loading GPT-2 model...")
model_name = "gpt2"  # Options: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Memory stats before inference
if torch.cuda.is_available():
    print(f"GPU memory allocated before inference: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Test GPT-2 generation
prompt = "The quick brown fox jumps over"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate text
start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

if start:
    start.record()

output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    do_sample=True
)

if end:
    end.record()
    torch.cuda.synchronize()
    print(f"Generation time: {start.elapsed_time(end):.2f} ms")

# Memory stats after inference
if torch.cuda.is_available():
    print(f"GPU memory allocated after inference: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated text:")
print("-" * 50)
print(generated_text)
print("-" * 50)

print("\nGPT-2 setup verification complete!")