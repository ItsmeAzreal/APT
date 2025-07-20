import torch

def handle_oom(error, step=None):
    if "out of memory" in str(error):
        print(f"OOM at step {step}! Skipping batch.")
        torch.cuda.empty_cache()
    else:
        raise error
