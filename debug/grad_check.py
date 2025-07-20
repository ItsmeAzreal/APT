import torch

def print_grad_norm(model, step=None):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    if step is not None:
        print(f"Step {step} | Grad Norm: {total_norm:.4f}")
    else:
        print(f"Grad Norm: {total_norm:.4f}")
