import torch

def print_sample_output(model, loader, tokenizer, step=None, device="cpu", n_samples=1, max_len=50):
    print(f"Generating {n_samples} sample outputs{' at step ' + str(step) if step is not None else ''}...")
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            for _ in range(n_samples):
                output = model(x)
                if hasattr(tokenizer, "decode"):
                    sample = output.argmax(dim=-1)[0]
                    decoded = tokenizer.decode(sample.tolist(), skip_special_tokens=True)
                    print(f"Sample {i}: {decoded[:max_len]}")
            if i == n_samples - 1:
                break
    model.train()
