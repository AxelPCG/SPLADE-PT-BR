#!/usr/bin/env python3
"""
File: upload_to_hf.py
Project: SPLADE-PT-BR
Created: Monday, 1st December 2025
Author: Axel

Purpose

  Upload SPLADE-PT-BR model to Hugging Face Hub.

  This script uploads the trained SPLADE model to HuggingFace Hub, including
  model checkpoint, configuration files, model card, and metadata.

Safety

  - Requires HF_TOKEN environment variable or .env file
  - Validates model checkpoint and required files before upload
  - Creates repository if it doesn't exist

Copyright (c) 2025 Axel. All rights reserved.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
import torch

# Load environment variables
load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "AxelPCG")
MODEL_NAME = "splade-pt-br"
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"

# Paths
PROJECT_ROOT = Path(__file__).parent
CHECKPOINT_DIR = PROJECT_ROOT / "splade" / "experiments" / "pt" / "checkpoint"
MODEL_CHECKPOINT = CHECKPOINT_DIR / "model_ckpt" / "model_final_checkpoint.tar"
CONFIG_FILE = CHECKPOINT_DIR / "config.yaml"
MODEL_CARD = PROJECT_ROOT / "model_card.md"
MODEL_METADATA = PROJECT_ROOT / "model_metadata.json"

# Temporary directory for preparing upload
UPLOAD_DIR = PROJECT_ROOT / "hf_upload_temp"


def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    if not HF_TOKEN or HF_TOKEN == "your_token_here":
        print("❌ ERROR: HF_TOKEN not set in .env file")
        print("   Please:")
        print("   1. Go to https://huggingface.co/settings/tokens")
        print("   2. Create a new token with 'write' permission")
        print("   3. Add it to .env file: HF_TOKEN=your_actual_token")
        return False
    
    if not MODEL_CHECKPOINT.exists():
        print(f"❌ ERROR: Model checkpoint not found at {MODEL_CHECKPOINT}")
        return False
    
    if not MODEL_CARD.exists():
        print(f"❌ ERROR: Model card not found at {MODEL_CARD}")
        return False
    
    print("✅ All requirements met!")
    return True


def prepare_upload_directory():
    """Prepare temporary directory with all files for upload"""
    print(f"\n📦 Preparing upload directory at {UPLOAD_DIR}...")
    
    # Clean and create upload directory
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir(parents=True)
    
    # Copy model checkpoint
    print("   Copying model checkpoint...")
    shutil.copy(MODEL_CHECKPOINT, UPLOAD_DIR / "pytorch_model.bin")
    
    # Copy config
    if CONFIG_FILE.exists():
        print("   Copying config.yaml...")
        shutil.copy(CONFIG_FILE, UPLOAD_DIR / "config.yaml")
    
    # Copy model card as README.md
    print("   Copying model card...")
    shutil.copy(MODEL_CARD, UPLOAD_DIR / "README.md")
    
    # Copy metadata
    if MODEL_METADATA.exists():
        print("   Copying metadata...")
        shutil.copy(MODEL_METADATA, UPLOAD_DIR / "model_metadata.json")
    
    # Create config.json for Hugging Face
    print("   Creating config.json...")
    config_json = {
        "architectures": ["Splade"],
        "model_type": "splade",
        "base_model": "neuralmind/bert-base-portuguese-cased",
        "vocab_size": 29794,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "aggregation": "max",
        "fp16": True
    }
    
    with open(UPLOAD_DIR / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)
    
    # Create tokenizer config
    print("   Creating tokenizer config...")
    tokenizer_config = {
        "tokenizer_class": "BertTokenizer",
        "do_lower_case": False,
        "model_max_length": 256,
        "tokenizer_type": "neuralmind/bert-base-portuguese-cased"
    }
    
    with open(UPLOAD_DIR / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # Create .gitattributes for Git LFS
    print("   Creating .gitattributes...")
    with open(UPLOAD_DIR / ".gitattributes", "w") as f:
        f.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.tar filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
    
    print("✅ Upload directory prepared!")
    return True


def create_repository():
    """Create HuggingFace repository"""
    print(f"\n🏗️  Creating repository {REPO_ID}...")
    
    try:
        api = HfApi(token=HF_TOKEN)
        
        # Try to create repo (will skip if already exists)
        repo_url = create_repo(
            repo_id=REPO_ID,
            token=HF_TOKEN,
            private=False,
            repo_type="model",
            exist_ok=True
        )
        
        print(f"✅ Repository ready: {repo_url}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR creating repository: {e}")
        return False


def upload_model():
    """Upload model files to HuggingFace"""
    print(f"\n📤 Uploading files to {REPO_ID}...")
    
    try:
        api = HfApi(token=HF_TOKEN)
        
        # Upload entire directory
        print("   Uploading all files (this may take several minutes)...")
        api.upload_folder(
            folder_path=str(UPLOAD_DIR),
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Upload SPLADE-PT-BR model v1.0.0"
        )
        
        print("✅ Upload completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR uploading files: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup():
    """Clean up temporary directory"""
    print("\n🧹 Cleaning up...")
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    print("✅ Cleanup complete!")


def main():
    """Main upload workflow"""
    print("=" * 70)
    print("🚀 SPLADE-PT-BR Hugging Face Upload Script")
    print("=" * 70)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Prepare files
    if not prepare_upload_directory():
        sys.exit(1)
    
    # Create repository
    if not create_repository():
        cleanup()
        sys.exit(1)
    
    # Upload model
    if not upload_model():
        cleanup()
        sys.exit(1)
    
    # Cleanup
    cleanup()
    
    # Success message
    print("\n" + "=" * 70)
    print("🎉 SUCCESS! Model uploaded to Hugging Face Hub!")
    print("=" * 70)
    print(f"\n📍 Model URL: https://huggingface.co/{REPO_ID}")
    print("\n✨ You can now use your model with:")
    print(f"   from splade.models.transformer_rep import Splade")
    print(f"   model = Splade.from_pretrained('{REPO_ID}')")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

