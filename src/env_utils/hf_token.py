# src/env_utils/hf_token.py

import os
from dotenv import load_dotenv

def load_hf_token():
    # Load .env file and set the token
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found in .env file!")
    os.environ["HF_TOKEN"] = hf_token
    try:
        from huggingface_hub import login
        login(hf_token)
    except ImportError:
        pass
    return hf_token
