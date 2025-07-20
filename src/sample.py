import torch
import re
import torch.nn.functional as F

def sample(model, start_context, length, temperature=1.0, top_k=None, device="cpu"):
    model.eval()
    result = start_context.clone().to(device)
    for _ in range(length):
        logits = model(result[:, -result.shape[1]:])  # Only last context
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            top_values, top_idx = torch.topk(logits, top_k)
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(1, top_idx, top_values)
            logits = mask
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        result = torch.cat((result, next_token), dim=1)
    return result

def decode_text(token_ids, tokenizer):
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    text = re.sub(r"([A-Z][A-Za-z\s]+:)", r"\n\1\n", text)
    text = re.sub(r"([.!?;])", r"\1\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()
