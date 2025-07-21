import torch
from transformers import LlamaTokenizer
from src.model import AnameeModel
from src.dataset import Curriculum2BTokenDataset

print("="*40)
print("DEBUG: LLM PIPELINE SANITY CHECK")
print("="*40)

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Tokenizer setup
try:
    # Change to your actual tokenizer name/path if needed
    tokenizer = LlamaTokenizer.from_pretrained("huggingface/llama-tokenizer")
    print("Tokenizer loaded successfully.")
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
except Exception as e:
    print("❌ Tokenizer load FAILED:")
    print(e)
    exit(1)

# Model setup
try:
    # Change model arguments if your class requires them
    model = AnameeModel(vocab_size=vocab_size)
    model.to(DEVICE)
    print("Model initialized and moved to device.")
except Exception as e:
    print("❌ Model initialization FAILED:")
    print(e)
    exit(1)

# Dataset & DataLoader setup
try:
    # Use a small subset for quick debug (add args if your class requires)
    train_dataset = Curriculum2BTokenDataset(tokenizer=tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,    # Tiny batch for debug
        shuffle=True,
        num_workers=0    # Avoids multi-process bugs on some systems
    )
    print("DataLoader created.")
except Exception as e:
    print("❌ DataLoader setup FAILED:")
    print(e)
    exit(1)

# Test: Run one batch through the model
try:
    model.eval()
    batch = next(iter(train_loader))
    x, y = batch
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    with torch.no_grad():
        output = model(x)
    print(f"Forward pass successful. Output shape: {output.shape}")
    print("✅ PIPELINE DEBUG: SUCCESS! You are ready to train.")
except Exception as e:
    print("❌ Pipeline test FAILED during forward pass:")
    print(e)
    exit(1)
