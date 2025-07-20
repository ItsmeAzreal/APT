# src/debug.py

import torch
import traceback
import sys

print("="*32)
print("DEBUG SCRIPT FOR PRETRAINING SETUP")
print("="*32)

# ---- 1. Imports from your main script
try:
    from train import train_loader, model, tokenizer, DEVICE, vocab_size
except Exception as e:
    print("Failed to import components from train.py")
    traceback.print_exc()
    sys.exit(1)

# ---- 2. Device/GPU Info
def print_device_info():
    print("\n[Device Info]")
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Device name:", torch.cuda.get_device_name(0))
        print("Memory allocated (MB):", torch.cuda.memory_allocated(0) / 1e6)
        print("Memory reserved (MB):", torch.cuda.memory_reserved(0) / 1e6)
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            print("GPU Temp (C):", temp)
            print("GPU Utilization (%):", util.gpu)
            pynvml.nvmlShutdown()
        except Exception as ee:
            print("pynvml not installed or failed:", ee)
    else:
        print("No CUDA device found.")

# ---- 3. DataLoader Check
def check_loader(train_loader):
    print("\n[Loader Check]")
    try:
        batch = next(iter(train_loader))
        if isinstance(batch, (list, tuple)):
            print("Batch shapes:", [x.shape for x in batch])
        else:
            print("Batch shape:", batch.shape)
    except Exception as e:
        print("Failed to get batch from train_loader:")
        traceback.print_exc()

# ---- 4. Forward/Backward/Gradient Check
def check_model_step(model, train_loader, device, vocab_size):
    print("\n[Model Forward/Backward/Grad Check]")
    model.train()
    try:
        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if logits.shape[:-1] != x.shape:
            print(f"WARNING: logits shape {logits.shape} does not match input shape {x.shape}")
        loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        print("Forward OK | Loss:", loss.item())
        loss.backward()
        grad_norm = 0.0
        for n, p in model.named_parameters():
            if p.grad is not None:
                grad_norm += (p.grad.data.norm(2) ** 2).item()
        grad_norm = grad_norm ** 0.5
        print("Grad norm:", grad_norm)
        if not torch.isfinite(loss):
            print("Loss is NOT finite!")
        else:
            print("Loss is finite.")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("OOM error during forward/backward pass! Try lowering batch size/block size.")
            torch.cuda.empty_cache()
        else:
            print("Error during forward/backward pass:")
            traceback.print_exc()
    except Exception as e:
        print("Error during model step:")
        traceback.print_exc()

# ---- 5. Sample Model Output
def print_sample_output(model, train_loader, tokenizer, device, vocab_size, n_samples=1, max_len=50):
    print("\n[Sample Output]")
    model.eval()
    try:
        x, _ = next(iter(train_loader))
        x = x.to(device)
        for i in range(n_samples):
            context = x[:1, :1]  # Use first token as context
            out = context
            for _ in range(max_len):
                logits = model(out)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                out = torch.cat([out, next_token], dim=1)
            decoded = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
            print(f"Sample {i+1}: {decoded[:200]}")
    except Exception as e:
        print("Sample output failed:")
        traceback.print_exc()

# ---- MAIN DEBUG FLOW ----
if __name__ == "__main__":
    print_device_info()
    check_loader(train_loader)
    check_model_step(model, train_loader, DEVICE, vocab_size)
    print_sample_output(model, train_loader, tokenizer, DEVICE, vocab_size, n_samples=2, max_len=100)

    print("\nALL DEBUG CHECKS COMPLETE. If all outputs look OK, you are ready for training.")
