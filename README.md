pip install --upgrade pip
pip install -r requirements.txt



🚀 LLM Pre-Training Pipeline
A full-featured, production-grade codebase for training Large Language Models (LLMs) from scratch, designed for reliability, cloud- or server-based runs, and maximum efficiency on big datasets.

📚 Table of Contents
Overview

Folder Structure

Quick Start

Environment Setup

Dataset Preparation

Configuration

Training Workflow

Monitoring & Logging

Checkpoints & Recovery

Common Issues & Tips

Best Practices

FAQ

-----------------------

🧠 Overview
This repository is for efficient, robust pre-training of LLMs using PyTorch, Hugging Face Transformers/Datasets, and distributed training.
It features:

Fast, streaming data pipeline with Hugging Face Datasets

Persistent local caching to avoid API/rate-limit errors

Curriculum learning (easy-to-hard)

Mixed-precision (AMP) and distributed (DDP) support

Safe, automatic checkpointing and resuming

Real-time validation, progress bars, and TensorBoard logging

RunPod/cloud friendly—works great on VS Code, Jupyter, or terminal

----------------

📁 Folder Structure

llm_pretrain/
├── data/               # (Optional) Local datasets, validation sets
├── scripts/            # Utility scripts (preprocessing, evaluation, etc.)
├── src/                # Source code (model, dataloader, train loop, etc.)
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── config.py
│   ├── utils.py
│   └── ...
├── checkpoints/        # Model checkpoints (created automatically)
├── runs/               # TensorBoard logs
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .env                # (Hidden) Hugging Face token and custom variables


---------------

🚀 Quick Start (RunPod or Local)

1. Clone the repo & open in VS Code/Jupyter/Terminal

git clone <your-repo-url>
cd llm_pretrain


2. [Optional but Recommended] Set up a Python virtual environment

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

4. Prepare your Hugging Face token and cache
Create a file named .env (never commit to GitHub!) and add:

HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

For best results and NO API 429 errors:

Use a persistent storage volume for your cache (RunPod: create and mount one at /workspace/hf_cache)

The code will automatically set cache paths—do not clear cache between runs.

5. Edit config as needed

Open src/config.py and set your batch size, block size, number of training steps, checkpoint interval, etc.

Set your RESUME_PATH to resume from a checkpoint, or leave blank for new runs.

----------------

📦 Dataset Preparation
Streaming from Hugging Face Datasets (default, recommended):

No need to pre-download. The code streams and caches shards as needed.

Change dataset/source in src/dataset.py.

Local/Custom Data:

Place files in data/ and adjust dataset.py as needed.

Preprocessing utilities in scripts/ (optional).

-------------

⚙️ Configuration
Edit src/config.py:

All main settings are at the top (batch, block size, learning rate, intervals, etc.)

Model architecture in src/model.py.

Comments explain every field.

BATCH_SIZE = 16
BLOCK_SIZE = 2048
TOTAL_TRAINING_STEPS = 6104  # 100M tokens (adjust as needed)
CHECKPOINT_INTERVAL = 3052   # Every 50M tokens
RESUME_PATH = ""             # Set to resume from a checkpoint


-------------

🏃 Training Workflow

A. Activate environment

source venv/bin/activate  # Or your conda env

B. Launch training (single or multi-GPU):

python src/train.py
# or for DDP:
torchrun --nproc_per_node=2 src/train.py


Training logs appear in the terminal and are saved to TensorBoard (runs/).

Checkpoints are auto-saved in checkpoints/ at your configured interval.

C. Resume training after stop/restart:

Upload your latest checkpoint to checkpoints/.

Set RESUME_PATH in src/config.py to the filename (e.g. checkpoints/full_ckpt_step3052.pt).

Re-launch training—script resumes automatically.

D. Each day or segment:

Download new checkpoint(s) from RunPod to your local machine for backup.

Upload and resume as needed for multi-day or segmented training.


--------------------

📊 Monitoring & Logging

TensorBoard:

tensorboard --logdir=runs/

Open the displayed URL in your browser.

Console logs: Train/val loss, steps, and checkpoints.

Progress bars: Training and validation use TQDM for real-time progress.

-----------------

💾 Checkpoints & Recovery
Checkpoints saved automatically every N steps/tokens (see CHECKPOINT_INTERVAL in config).

Best practice: Download checkpoints after every session for backup.

To resume: Place checkpoint in checkpoints/, set RESUME_PATH in config, and restart training.

-----------------

⚠️ Common Issues & Tips
Python Version: Use 3.10 or 3.11. Do not use 3.13+ (many ML libraries incompatible).

CUDA Errors: Ensure your PyTorch and CUDA versions match your hardware. Use official wheels.

Out of Memory: Lower BATCH_SIZE or BLOCK_SIZE. Use nvidia-smi to monitor GPU usage.

Slow Download/Streaming: Use a persistent Hugging Face cache. If on RunPod, mount a Storage Volume at /workspace/hf_cache.

429 (Rate Limit from Hugging Face): Use persistent cache, limit number of concurrent workers, and never clear cache between runs. Your .env HF token is picked up automatically.

Checkpoints/Storage Full: Periodically clean out old checkpoints and models from checkpoints/ and cache to avoid running out of disk space (see below for clean-up tips).

----------------

🌟 Best Practices
Test with small configs before scaling up to large training runs.

Always validate your dataset and data pipeline before long runs.

Set up and monitor TensorBoard.

Keep requirements and Python versions fixed for full reproducibility.

Download and back up critical checkpoints after each session.


---------------

🗑️ Cleaning Up Old Cache or Checkpoints

f you’re low on space, in your RunPod (or local) terminal:

# Check usage
du -sh /workspace/hf_cache
du -sh /workspace/checkpoints

# Delete a specific model/dataset from cache
rm -rf /workspace/hf_cache/models--UnneededModelName
rm -rf /workspace/hf_cache/datasets/UnneededDatasetName

# Delete old checkpoints (keep latest N)
ls -tr /workspace/checkpoints | head -n -3 | xargs -I {} rm /workspace/checkpoints/{}

-----------------

❓ FAQ
Q: What if my cache gets wiped after RunPod shutdown?
A: Always download your checkpoints after each session. For best results, mount your HF cache to a persistent volume—otherwise, you will have to re-download data/models after every restart.

Q: How do I avoid Hugging Face API rate limits (429)?
A: Use a persistent cache, never clear it, and limit parallel workers. Your pipeline is set up for this already.

Q: Can I safely stop and resume multi-day training?
A: Yes! Just save/download your checkpoint, upload it next time, and set RESUME_PATH to resume.

Q: Can I use more than 2 GPUs?
A: Yes, code supports multi-GPU via DDP—adjust your torchrun command.

Q: How can I monitor disk usage?
A: In your pod terminal:
df -h /workspace
du -sh /workspace/hf_cache
du -sh /workspace/checkpoints

Q: Is my training quality affected by cache or resume?
A: No, cache only affects download speed. Resuming from checkpoints is perfectly safe and preserves all optimizer/model state.

------------------

🤖 Techniques Used
Distributed Data Parallel (DDP) training

Mixed precision (AMP) with bfloat16/float16

Streaming + sharded dataset support (curriculum)

Automatic checkpointing & robust resume

Persistent Hugging Face caching

TensorBoard and console progress/logging

Validation and sampling pipeline for monitoring

Safe gradient clipping & OOM handling

Supports Hugging Face tokens from .env


--------------------

📝 Extra: Troubleshooting & Tips
For "CUDA out of memory" errors: lower batch/block size, or check for memory leaks.

For "429 Too Many Requests" errors: ensure persistent cache, fewer workers, single download per pod.

For "File not found" on checkpoints: double-check paths and file uploads after pod restart.

For debugging: use NUM_WORKERS=0 for streaming datasets, increase only for local datasets.



