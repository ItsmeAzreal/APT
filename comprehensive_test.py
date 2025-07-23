#!/usr/bin/env python3
# debug/comprehensive_test.py - Comprehensive system test for the training pipeline

import os
import sys
import torch
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment():
    """Test environment setup"""
    print("\n" + "="*60)
    print("🔍 ENVIRONMENT TEST")
    print("="*60)
    
    # Load environment
    load_dotenv(dotenv_path=".env")
    
    # Check Python version
    python_version = sys.version_info
    print(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major != 3 or python_version.minor < 8:
        print("❌ Python 3.8+ required!")
        return False
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1e9:.1f}GB")
    
    # Check environment variables
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found in .env file!")
        return False
    print("✓ HF_TOKEN found")
    
    # Check directories
    dirs = ["checkpoints", "runs", "logs"]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✓ Created directory: {dir_name}")
        else:
            print(f"✓ Directory exists: {dir_name}")
    
    return True

def test_imports():
    """Test all required imports"""
    print("\n" + "="*60)
    print("📦 IMPORT TEST")
    print("="*60)
    
    required_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("tqdm", "TQDM"),
        ("tensorboard", "TensorBoard"),
        ("dotenv", "Python-dotenv"),
    ]
    
    optional_imports = [
        ("flash_attn", "Flash Attention"),
        ("bitsandbytes", "BitsAndBytes"),
        ("wandb", "Weights & Biases"),
    ]
    
    all_good = True
    
    # Test required imports
    for module, name in required_imports:
        try:
            __import__(module)
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {name}: {e}")
            all_good = False
    
    # Test optional imports
    print("\nOptional packages:")
    for module, name in optional_imports:
        try:
            __import__(module)
            print(f"✓ {name} available")
        except ImportError:
            print(f"⚠️  {name} not available (optional)")
    
    return all_good

def test_config():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("⚙️  CONFIGURATION TEST")
    print("="*60)
    
    try:
        from src import config
        
        critical_vars = [
            "BATCH_SIZE", "BLOCK_SIZE", "LEARNING_RATE",
            "TOTAL_TOKENS_TARGET", "MODEL_DIM", "NUM_LAYERS"
        ]
        
        for var in critical_vars:
            if hasattr(config, var):
                value = getattr(config, var)
                print(f"✓ {var}: {value}")
            else:
                print(f"❌ Missing config: {var}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False

def test_model():
    """Test model initialization"""
    print("\n" + "="*60)
    print("🤖 MODEL TEST")
    print("="*60)
    
    try:
        from src.model import AnameeModel, count_parameters
        from src.config import *
        
        # Initialize model with small config for testing
        model = AnameeModel(
            vocab_size=32000,
            dim=256,  # Small for testing
            num_heads=4,
            hidden_dim=1024,
            num_layers=4,  # Few layers for testing
            num_kv_heads=2,
            max_seq_len=512,
            dropout=0.0
        )
        
        # Test forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Create dummy input
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(0, 32000, (batch_size, seq_len)).to(device)
        labels = torch.randint(0, 32000, (batch_size, seq_len)).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
        
        print(f"✓ Model initialized successfully")
        print(f"✓ Parameters: {count_parameters(model):,}")
        print(f"✓ Forward pass successful")
        print(f"✓ Loss: {outputs.loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tokenizer():
    """Test tokenizer loading"""
    print("\n" + "="*60)
    print("📝 TOKENIZER TEST")
    print("="*60)
    
    try:
        from transformers import LlamaTokenizer
        from src.config import TOKENIZER_NAME, HF_CACHE_DIR
        
        tokenizer = LlamaTokenizer.from_pretrained(
            TOKENIZER_NAME,
            token=os.getenv("HF_TOKEN"),
            cache_dir=HF_CACHE_DIR
        )
        
        # Test tokenization
        test_text = "Hello, this is a test of the tokenizer!"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        print(f"✓ Tokenizer loaded: {TOKENIZER_NAME}")
        print(f"✓ Vocab size: {tokenizer.vocab_size}")
        print(f"✓ Test text: '{test_text}'")
        print(f"✓ Token count: {len(tokens)}")
        print(f"✓ Decoded: '{decoded}'")
        
        return True
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False

def test_dataset():
    """Test dataset loading (minimal)"""
    print("\n" + "="*60)
    print("📊 DATASET TEST")
    print("="*60)
    
    try:
        from transformers import LlamaTokenizer
        from src.dataset import Curriculum2BTokenDataset
        from src.config import TOKENIZER_NAME, HF_CACHE_DIR, BLOCK_SIZE
        
        # Load tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            TOKENIZER_NAME,
            token=os.getenv("HF_TOKEN"),
            cache_dir=HF_CACHE_DIR
        )
        
        # Create dataset with tiny targets for testing
        dataset = Curriculum2BTokenDataset(
            tokenizer=tokenizer,
            block_size=BLOCK_SIZE,
            easy_token_target=1000,  # Tiny for testing
            hard_token_target=1000,  # Tiny for testing
            seed=42,
            shard_id=0,
            num_shards=1,
            cache_dir=HF_CACHE_DIR
        )
        
        print("✓ Dataset created successfully")
        print("⏳ Testing iteration (this may take a moment)...")
        
        # Test iteration (just one batch)
        data_iter = iter(dataset)
        try:
            x, y = next(data_iter)
            print(f"✓ First batch shape: x={x.shape}, y={y.shape}")
            print(f"✓ Data types: x={x.dtype}, y={y.dtype}")
        except StopIteration:
            print("⚠️  No data retrieved (might be filtering issue)")
        
        return True
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory():
    """Test GPU memory requirements"""
    print("\n" + "="*60)
    print("💾 MEMORY TEST")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠️  No GPU available, skipping memory test")
        return True
    
    try:
        from src.model import AnameeModel
        from src.config import *
        
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        
        # Get initial memory
        initial_memory = torch.cuda.memory_allocated(device) / 1e9
        print(f"Initial GPU memory: {initial_memory:.2f}GB")
        
        # Create model
        model = AnameeModel(
            vocab_size=VOCAB_SIZE,
            dim=MODEL_DIM,
            num_heads=NUM_HEADS,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS,
            max_seq_len=BLOCK_SIZE
        ).to(device)
        
        model_memory = torch.cuda.memory_allocated(device) / 1e9
        print(f"Model memory: {model_memory:.2f}GB")
        
        # Test forward pass with batch
        batch = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE)).to(device)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(batch)
        
        peak_memory = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"Peak memory during forward: {peak_memory:.2f}GB")
        
        # Check if we have enough memory
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        memory_usage_pct = (peak_memory / total_memory) * 100
        
        print(f"Total GPU memory: {total_memory:.2f}GB")
        print(f"Memory usage: {memory_usage_pct:.1f}%")
        
        if memory_usage_pct > 90:
            print("⚠️  High memory usage! Consider reducing batch size.")
        else:
            print("✓ Memory usage is within safe limits")
        
        return True
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🚀 COMPREHENSIVE SYSTEM TEST FOR LLM PRETRAINING")
    print("="*80)
    
    tests = [
        ("Environment", test_environment),
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Model", test_model),
        ("Tokenizer", test_tokenizer),
        ("Dataset", test_dataset),
        ("Memory", test_memory),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("📋 TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED! System is ready for training.")
    else:
        print("⚠️  Some tests failed. Please fix the issues before training.")
        sys.exit(1)

if __name__ == "__main__":
    main()