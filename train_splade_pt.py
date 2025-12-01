#!/usr/bin/env python3
"""
SPLADE v2 PT-BR - Training Script

This script provides a complete workflow for training the SPLADE model for Portuguese (PT-BR) text retrieval.
It is a modularized version of the training notebook.

Usage:
    python train_splade_pt.py [--skip-setup] [--skip-download] [--skip-config] [--skip-qrel]
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Optional

# Third-party imports
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("❌ Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
    sys.exit(1)


# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
SPLADE_DIR = PROJECT_ROOT / "splade"
DATA_DIR = PROJECT_ROOT / "data"


def setup_repository_and_patch() -> bool:
    """
    Clone SPLADE repository if it doesn't exist and apply compatibility patches.
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("=" * 80)
    print("Step 1: Setup Repository and Apply Compatibility Patches")
    print("=" * 80)
    
    # Clone repository if it doesn't exist
    if not SPLADE_DIR.exists():
        print("📦 Cloning SPLADE repository...")
        result = os.system(f"cd {PROJECT_ROOT} && git clone https://github.com/leobavila/splade.git")
        if result != 0:
            print("❌ Error: Failed to clone repository")
            return False
        print("✅ Repository cloned")
    else:
        print("✅ Repository already exists")
    
    # Apply compatibility patch for AdamW optimizer
    file_path = SPLADE_DIR / "splade" / "optim" / "bert_optim.py"
    if not file_path.exists():
        print("❌ Error: bert_optim.py file not found")
        return False
    
    with open(file_path, "r") as f:
        content = f.read()
    
    # Check if patch already applied
    if "from torch.optim import AdamW" in content:
        print("✅ Patch already applied")
        return True
    
    # Apply patch
    new_content = content.replace(
        "from transformers.optimization import AdamW, get_linear_schedule_with_warmup",
        "from transformers import get_linear_schedule_with_warmup; from torch.optim import AdamW"
    )
    
    with open(file_path, "w") as f:
        f.write(new_content)
    print("✅ Patch applied: bert_optim.py fixed")
    
    return True


def download_from_hf(repo_id: str, filename: str, output_path: Path, description: str) -> bool:
    """
    Download file from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        filename: File name in the repository
        output_path: Local output path
        description: Description for logging
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Skip if file already exists and has valid size (> 100 bytes)
    if output_path.exists() and output_path.stat().st_size > 100:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ {description} already exists ({size_mb:.1f} MB), skipping download.")
        return True
    
    print(f"📥 Downloading {description}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=None
        )
        # Copy to desired destination
        shutil.copy(downloaded_path, output_path)
        if output_path.exists() and output_path.stat().st_size > 100:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ {description} downloaded successfully ({size_mb:.1f} MB)")
            return True
        else:
            print(f"❌ {description} failed: file too small or empty")
            return False
    except Exception as e:
        print(f"❌ Error downloading {description}: {e}")
        return False


def download_datasets() -> bool:
    """
    Download and prepare training and validation datasets.
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("=" * 80)
    print("Step 2: Download and Prepare Datasets")
    print("=" * 80)
    print("⏳ Downloading public datasets from HuggingFace Hub... (This may take several minutes)")
    
    # Create base directories
    mmarco_dir = DATA_DIR / "m_marco"
    mrobust_dir = DATA_DIR / "m_robust"
    mmarco_dir.mkdir(parents=True, exist_ok=True)
    mrobust_dir.mkdir(parents=True, exist_ok=True)
    
    # Create SPLADE destination directories
    (SPLADE_DIR / "data" / "pt" / "triplets").mkdir(parents=True, exist_ok=True)
    (SPLADE_DIR / "data" / "pt" / "val_retrieval" / "collection").mkdir(parents=True, exist_ok=True)
    (SPLADE_DIR / "data" / "pt" / "val_retrieval" / "queries").mkdir(parents=True, exist_ok=True)
    
    # --- mMARCO (Training Dataset) ---
    print("\n📦 Downloading mMARCO datasets...")
    mmarco_files = {
        "queries_train.tsv": ("data/google/queries/train/portuguese_queries.train.tsv", "queries_train.tsv"),
        "corpus.tsv": ("data/google/collections/portuguese_collection.tsv", "corpus.tsv"),
        "triples.train.ids.small.tsv": ("data/triples.train.ids.small.tsv", "triples.train.ids.small.tsv")
    }
    
    success = True
    for local_name, (remote_path, description) in mmarco_files.items():
        output_path = mmarco_dir / local_name
        if not download_from_hf("unicamp-dl/mmarco", remote_path, output_path, description):
            success = False
    
    # Verify and copy mMARCO files
    if success and all((mmarco_dir / f).exists() and (mmarco_dir / f).stat().st_size > 100 
                       for f in mmarco_files.keys()):
        shutil.copy(mmarco_dir / "corpus.tsv", SPLADE_DIR / "data" / "pt" / "triplets" / "corpus.tsv")
        shutil.copy(mmarco_dir / "queries_train.tsv", SPLADE_DIR / "data" / "pt" / "triplets" / "queries_train.tsv")
        shutil.copy(mmarco_dir / "triples.train.ids.small.tsv", SPLADE_DIR / "data" / "pt" / "triplets" / "raw.tsv")
        print("✅ mMARCO files copied to SPLADE structure")
    else:
        print("❌ Error: Some mMARCO files were not downloaded correctly")
        return False
    
    # --- mRobust (Validation Dataset) ---
    print("\n📦 Downloading mRobust datasets...")
    mrobust_files = {
        "queries.tsv": ("data/queries/portuguese_queries.tsv", "mrobust queries.tsv"),
        "corpus.tsv": ("data/collections/portuguese_collection.tsv", "mrobust corpus.tsv"),
        "qrels.robust04.txt": ("qrels.robust04.txt", "qrels.robust04.txt")
    }
    
    success = True
    for local_name, (remote_path, description) in mrobust_files.items():
        output_path = mrobust_dir / local_name
        if not download_from_hf("unicamp-dl/mrobust", remote_path, output_path, description):
            success = False
    
    # Verify and copy mRobust files
    if success and all((mrobust_dir / f).exists() and (mrobust_dir / f).stat().st_size > 100 
                       for f in mrobust_files.keys()):
        shutil.copy(mrobust_dir / "corpus.tsv", SPLADE_DIR / "data" / "pt" / "val_retrieval" / "collection" / "raw.tsv")
        shutil.copy(mrobust_dir / "queries.tsv", SPLADE_DIR / "data" / "pt" / "val_retrieval" / "queries" / "raw.tsv")
        print("✅ mRobust files copied to SPLADE structure")
    else:
        print("❌ Error: Some mRobust files were not downloaded correctly")
        return False
    
    print("\n✅ Download process completed.")
    return True


def convert_qrels_to_json() -> bool:
    """
    Convert TREC-format QRELS file to JSON format.
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("=" * 80)
    print("Step 3: Convert QRELS to JSON")
    print("=" * 80)
    
    qrel_path = DATA_DIR / "m_robust" / "qrels.robust04.txt"
    output_path = SPLADE_DIR / "data" / "pt" / "val_retrieval" / "qrel.json"
    
    if not qrel_path.exists():
        print(f"❌ Error: qrels.robust04.txt not found at {qrel_path}")
        return False
    
    # Convert TREC-format QRELS to JSON
    # Format: query_id 0 doc_id relevance_score
    qrel = defaultdict(dict)
    
    with open(qrel_path, 'r') as file:
        for line in file:
            fields = line.split()
            if len(fields) >= 4:
                q_id = fields[0]
                doc_id = fields[2]
                rel = fields[3]
                qrel[q_id][doc_id] = int(rel)
    
    # Save as JSON for SPLADE evaluation
    with open(output_path, 'w') as file:
        json.dump(qrel, file)
    
    print(f"✅ QREL converted to JSON: {output_path}")
    print(f"   Total queries: {len(qrel)}")
    return True


def create_configuration_files() -> bool:
    """
    Generate all Hydra configuration files required for training.
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("=" * 80)
    print("Step 4: Generate Configuration Files")
    print("=" * 80)
    
    # Create directory structure
    conf_dirs = [
        SPLADE_DIR / "conf" / "train" / "config",
        SPLADE_DIR / "conf" / "train" / "data",
        SPLADE_DIR / "conf" / "train" / "model",
        SPLADE_DIR / "conf" / "index",
        SPLADE_DIR / "conf" / "retrieve_evaluate",
        SPLADE_DIR / "conf" / "flops"
    ]
    
    for conf_dir in conf_dirs:
        conf_dir.mkdir(parents=True, exist_ok=True)
    
    # Model Configuration
    model_config_path = SPLADE_DIR / "conf" / "train" / "model" / "splade_bertimbau_base.yaml"
    with open(model_config_path, "w") as f:
        f.write("""_target_: splade.models.transformer_rep.Splade
# Note: The actual parameter will be read from init_dict below
model_type_or_dir: neuralmind/bert-base-portuguese-cased
""")
    
    # Data Configuration
    data_config_path = SPLADE_DIR / "conf" / "train" / "data" / "pt.yaml"
    with open(data_config_path, "w") as f:
        f.write(f"""# @package _global_
data:
    type: triplets
    TRAIN_DATA_DIR: {SPLADE_DIR}/data/pt/triplets
    VALIDATION_DATA_DIR: {SPLADE_DIR}/data/pt/val_retrieval
    QREL_PATH: {SPLADE_DIR}/data/pt/val_retrieval/qrel.json
""")
    
    # Training Configuration
    train_config_path = SPLADE_DIR / "conf" / "train" / "config" / "splade_pt.yaml"
    with open(train_config_path, "w") as f:
        f.write("""# @package _global_
config:
    lr: 2e-5
    seed: 123
    gradient_accumulation_steps: 1
    weight_decay: 0.01
    validation_metrics: [MRR@10]
    pretrained_no_yaml_config: false
    nb_iterations: 150000
    train_batch_size: 32
    eval_batch_size: 32
    index_retrieval_batch_size: 32
    record_frequency: 1000
    train_monitoring_freq: 500
    warmup_steps: 6000
    max_length: 256
    fp16: true
    matching_type: splade
    monitoring_ckpt: true
    tokenizer_type: neuralmind/bert-base-portuguese-cased

    # Required loss parameter (fixes ConfigKeyError)
    loss: InBatchPairwiseNLL

    # Required keys for Hydra (will be overridden at runtime)
    checkpoint_dir: ""
    index_dir: ""
    out_dir: ""

    regularization:
        FLOPS:
            lambda_q: 0.0003
            lambda_d: 0.0001
            T: 50000
""")
    
    # Main Configuration
    main_config_path = SPLADE_DIR / "conf" / "config_splade_pt.yaml"
    with open(main_config_path, "w") as f:
        f.write("""defaults:
  - train/data: pt
  - train/model: splade_bertimbau_base
  - train/config: splade_pt
  - index: pt
  - retrieve_evaluate: pt
  - flops: pt
  - _self_

# init_dict with previous corrections
init_dict:
  model_type_or_dir: neuralmind/bert-base-portuguese-cased
  fp16: true

hydra:
  run:
    dir: experiments/pt/out
  job:
    chdir: true
""")
    
    # Placeholder Configurations
    placeholders = {
        SPLADE_DIR / "conf" / "index" / "pt.yaml": "# Placeholder",
        SPLADE_DIR / "conf" / "retrieve_evaluate" / "pt.yaml": "# Placeholder",
        SPLADE_DIR / "conf" / "flops" / "pt.yaml": "# Placeholder"
    }
    
    for path, content in placeholders.items():
        with open(path, "w") as f:
            f.write(content)
    
    print("✅ Configuration files created successfully (loss: InBatchPairwiseNLL included).")
    return True


def run_training() -> bool:
    """
    Execute SPLADE model training.
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("=" * 80)
    print("Step 5: Execute Training")
    print("=" * 80)
    
    # Configure environment
    pythonpath = os.environ.get('PYTHONPATH', '')
    if str(SPLADE_DIR) not in pythonpath:
        os.environ['PYTHONPATH'] = f"{pythonpath}:{SPLADE_DIR}" if pythonpath else str(SPLADE_DIR)
    
    os.environ['SPLADE_CONFIG_NAME'] = "config_splade_pt.yaml"
    os.environ['PYTHONUNBUFFERED'] = '1'  # Disable buffering to see real-time logs
    
    print("🚀 Starting training... Follow the logs below.")
    print("Note: Ignore warnings about 'Unable to register cuFFT/cuDNN' from TensorFlow/JAX.")
    print("=" * 80)
    print()
    
    # Execute training with real-time output
    cmd = [
        sys.executable,
        '-m', 'splade.train_from_triplets_ids',
        'config.checkpoint_dir=experiments/pt/checkpoint',
        'config.index_dir=experiments/pt/index',
        'config.out_dir=experiments/pt/out'
    ]
    
    # Change to splade directory
    original_dir = os.getcwd()
    os.chdir(SPLADE_DIR)
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,  # Line buffered
            env=os.environ.copy()
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\n❌ Training finished with exit code: {process.returncode}")
            return False
        else:
            print("\n✅ Training completed successfully!")
            print("\n📁 Checkpoints saved in: experiments/pt/checkpoint/")
            print("📊 Training logs in: experiments/pt/out/")
            return True
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
        print("💡 Training can be resumed from the last checkpoint.")
        if 'process' in locals():
            process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_dir)


def main():
    """Main function to orchestrate the training workflow."""
    parser = argparse.ArgumentParser(
        description="Train SPLADE model for Portuguese text retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training workflow
  python train_splade_pt.py
  
  # Skip setup (repository already cloned and patched)
  python train_splade_pt.py --skip-setup
  
  # Skip download (datasets already downloaded)
  python train_splade_pt.py --skip-download
  
  # Only run training (everything else already done)
  python train_splade_pt.py --skip-setup --skip-download --skip-config --skip-qrel
        """
    )
    
    parser.add_argument(
        '--skip-setup',
        action='store_true',
        help='Skip repository setup and patching'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip dataset download'
    )
    parser.add_argument(
        '--skip-qrel',
        action='store_true',
        help='Skip QREL conversion'
    )
    parser.add_argument(
        '--skip-config',
        action='store_true',
        help='Skip configuration file generation'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training execution (only prepare environment)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SPLADE v2 PT-BR - Training Script")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"SPLADE directory: {SPLADE_DIR}")
    print()
    
    # Step 1: Setup repository and apply patches
    if not args.skip_setup:
        if not setup_repository_and_patch():
            print("\n❌ Setup failed. Exiting.")
            sys.exit(1)
        print()
    
    # Step 2: Download datasets
    if not args.skip_download:
        if not download_datasets():
            print("\n❌ Dataset download failed. Exiting.")
            sys.exit(1)
        print()
    
    # Step 3: Convert QRELS
    if not args.skip_qrel:
        if not convert_qrels_to_json():
            print("\n❌ QREL conversion failed. Exiting.")
            sys.exit(1)
        print()
    
    # Step 4: Generate configuration files
    if not args.skip_config:
        if not create_configuration_files():
            print("\n❌ Configuration generation failed. Exiting.")
            sys.exit(1)
        print()
    
    # Step 5: Run training
    if not args.skip_training:
        if not run_training():
            print("\n❌ Training failed.")
            sys.exit(1)
    else:
        print("✅ Environment prepared. Training skipped (use without --skip-training to train).")
    
    print("\n" + "=" * 80)
    print("✅ All steps completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

