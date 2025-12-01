#!/usr/bin/env python3
"""
File: compare_models.py
Project: SPLADE-PT-BR
Created: Monday, 1st December 2025
Author: Axel

Purpose

  Compare SPLADE-PT-BR model with original SPLADE++ model benchmarks.

  This script compares evaluation metrics between the Portuguese-trained model
  and the original English SPLADE++ model, providing performance comparisons
  and generating a detailed comparison report.

Copyright (c) 2025 Axel. All rights reserved.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

# Original SPLADE++ benchmarks (MS MARCO English)
ORIGINAL_SPLADE_METRICS = {
    "model_name": "SPLADE++ (Original)",
    "language": "English",
    "dataset": "MS MARCO",
    "metrics": {
        "MRR@10": 0.368,
        "Recall@100": 0.962,
        "Recall@1000": 0.995
    },
    "sparsity": {
        "avg_query_L0": 120,
        "avg_doc_L0": 150
    }
}

# Paths (script is in scripts/utils/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
SPLADE_DIR = PROJECT_ROOT / "splade"
RESULTS_DIR = SPLADE_DIR / "experiments" / "pt" / "out"
PERF_FILE = RESULTS_DIR / "perf_all_datasets.json"
OUTPUT_FILE = RESULTS_DIR / "comparison_report.json"


def load_evaluation_results() -> Dict[str, Any]:
    """Load evaluation results from the Portuguese model"""
    if not PERF_FILE.exists():
        print(f"⚠️  Warning: Evaluation results not found at {PERF_FILE}")
        print("   Please run the evaluation pipeline first:")
        print("   cd splade && python -m splade.index +config=config_splade_pt")
        print("   cd splade && python -m splade.retrieve +config=config_splade_pt")
        return None
    
    with open(PERF_FILE, 'r') as f:
        results = json.load(f)
    
    return results


def format_metric_value(value):
    """Format metric value for display"""
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def calculate_improvement(original, current):
    """Calculate percentage improvement"""
    if original == 0:
        return "N/A"
    improvement = ((current - original) / original) * 100
    return f"{improvement:+.2f}%"


def compare_metrics():
    """Compare metrics between models"""
    print("=" * 80)
    print("📊 SPLADE-PT-BR vs Original SPLADE++ Comparison")
    print("=" * 80)
    
    # Load results
    pt_results = load_evaluation_results()
    
    if pt_results is None:
        print("\n❌ Cannot perform comparison without evaluation results.")
        print("   Run evaluation first, then rerun this script.")
        return
    
    # Prepare comparison data
    comparison = {
        "original_model": ORIGINAL_SPLADE_METRICS,
        "splade_pt_br": {
            "model_name": "SPLADE-PT-BR",
            "language": "Portuguese (Brazilian)",
            "dataset": "mMARCO Portuguese",
            "metrics": {},
            "raw_results": pt_results
        },
        "comparison": {}
    }
    
    # Extract metrics from PT results
    # The results might be nested by dataset name
    if isinstance(pt_results, dict):
        # Find first dataset results
        dataset_key = list(pt_results.keys())[0] if pt_results else None
        if dataset_key:
            metrics = pt_results[dataset_key]
            comparison["splade_pt_br"]["metrics"] = metrics
    
    # Print comparison table
    print("\n📈 Performance Metrics Comparison")
    print("-" * 80)
    print(f"{'Metric':<20} {'Original (EN)':<20} {'SPLADE-PT (PT)':<20} {'Difference':<15}")
    print("-" * 80)
    
    pt_metrics = comparison["splade_pt_br"]["metrics"]
    
    for metric_name, original_value in ORIGINAL_SPLADE_METRICS["metrics"].items():
        # Try to find corresponding metric in PT results
        pt_value = None
        
        # Try exact match first
        if metric_name in pt_metrics:
            pt_value = pt_metrics[metric_name]
        else:
            # Try variations (e.g., "MRR@10" vs "mrr_10")
            metric_variations = [
                metric_name,
                metric_name.lower(),
                metric_name.replace("@", "_"),
                metric_name.lower().replace("@", "_")
            ]
            for var in metric_variations:
                if var in pt_metrics:
                    pt_value = pt_metrics[var]
                    break
        
        if pt_value is not None:
            diff = calculate_improvement(original_value, pt_value)
            comparison["comparison"][metric_name] = {
                "original": original_value,
                "splade_pt_br": pt_value,
                "difference": diff
            }
        else:
            diff = "N/A"
            comparison["comparison"][metric_name] = {
                "original": original_value,
                "splade_pt_br": "N/A",
                "difference": "N/A"
            }
        
        pt_display = format_metric_value(pt_value) if pt_value is not None else "N/A"
        print(f"{metric_name:<20} {format_metric_value(original_value):<20} {pt_display:<20} {diff:<15}")
    
    # Print additional metrics from PT model
    print("\n📊 Additional Portuguese Model Metrics:")
    print("-" * 80)
    for key, value in pt_metrics.items():
        if key.upper() not in ORIGINAL_SPLADE_METRICS["metrics"]:
            print(f"  {key}: {format_metric_value(value)}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("📝 Analysis")
    print("=" * 80)
    
    print("\n✅ Key Observations:")
    print("   • Direct comparison is limited due to different datasets")
    print("   • Original SPLADE++ trained on MS MARCO (English)")
    print("   • SPLADE-PT-BR trained on mMARCO (Portuguese)")
    print("   • Language-specific vocabulary and semantic patterns")
    
    print("\n🎯 Advantages of SPLADE-PT-BR for Portuguese:")
    print("   1. Native Portuguese vocabulary (no subword splitting)")
    print("   2. Contextual expansion with Portuguese linguistic patterns")
    print("   3. Better semantic understanding of Brazilian Portuguese")
    print("   4. Optimized for Portuguese retrieval tasks")
    
    # Performance guidance
    print("\n📏 Performance Guidance:")
    if any(k in pt_metrics for k in ["MRR@10", "mrr_10"]):
        mrr_value = pt_metrics.get("MRR@10") or pt_metrics.get("mrr_10")
        if mrr_value:
            if mrr_value > 0.35:
                print("   🏆 Excellent: MRR@10 > 0.35 (State-of-the-art)")
            elif mrr_value > 0.30:
                print("   ✅ Very Good: MRR@10 > 0.30")
            elif mrr_value > 0.25:
                print("   👍 Good: MRR@10 > 0.25")
            else:
                print("   ⚠️  Fair: MRR@10 < 0.25 (May benefit from more training)")
    
    # Save comparison report
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Full comparison report saved to: {OUTPUT_FILE}")
    print("=" * 80)


def main():
    """Main comparison workflow"""
    try:
        compare_metrics()
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

