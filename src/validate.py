import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

@torch.no_grad()
def validate(model, val_loader, device, val_steps=100):
    model.eval()
    total_loss = 0
    val_iter = iter(val_loader)
    pbar = tqdm(range(val_steps), desc="Validation", leave=False)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    for _ in pbar:
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        with autocast(device.type, dtype=torch.bfloat16 if use_bf16 else torch.float16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    model.train()
    return total_loss / pbar.n if pbar.n > 0 else 0.0
