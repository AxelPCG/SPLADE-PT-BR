#!/usr/bin/env python3
"""
SPLADE-PT-BR Evaluation & Comparison Pipeline

This script runs comprehensive evaluation with model comparison:
1. Index documents for both SPLADE-PT-BR and SPLADE-EN
2. Retrieve and calculate metrics for both models
3. Generate detailed comparison reports
4. Create comparative visualizations

Usage:
    python scripts/evaluation/run_evaluation_comparator.py [options]
    
Options:
    --skip-index-pt        Skip indexing for SPLADE-PT-BR
    --skip-index-en        Skip indexing for SPLADE-EN
    --skip-retrieve-pt     Skip retrieval for SPLADE-PT-BR
    --skip-retrieve-en     Skip retrieval for SPLADE-EN
    --skip-visualization   Skip visualization generation
    --pt-only              Only evaluate SPLADE-PT-BR (no comparison)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
SPLADE_DIR = PROJECT_ROOT / "splade"
RESULTS_DIR = SPLADE_DIR / "experiments"
EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"


class ProgressTracker:
    """Track and display evaluation progress"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_times = []
        self.start_time = time.time()
        self.step_start_time = None
        
    def start_step(self, step_name: str):
        """Start tracking a new step"""
        self.current_step += 1
        self.step_start_time = time.time()
        
        # Calculate progress
        progress = (self.current_step - 1) / self.total_steps * 100
        bar_length = 40
        filled = int(bar_length * (self.current_step - 1) / self.total_steps)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Calculate ETA
        elapsed = time.time() - self.start_time
        if self.current_step > 1:
            avg_time = elapsed / (self.current_step - 1)
            remaining_steps = self.total_steps - (self.current_step - 1)
            eta_seconds = avg_time * remaining_steps
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."
        
        print("\n" + "=" * 80)
        print(f"📊 Progress: [{bar}] {progress:.1f}% | Step {self.current_step}/{self.total_steps}")
        print(f"⏱️  Elapsed: {self._format_time(elapsed)} | ETA: {eta_str}")
        print(f"🔄 Current: {step_name}")
        print("=" * 80 + "\n")
    
    def end_step(self, success: bool):
        """End tracking current step"""
        if self.step_start_time:
            duration = time.time() - self.step_start_time
            self.step_times.append(duration)
            status = "✅" if success else "❌"
            print(f"\n{status} Step completed in {self._format_time(duration)}")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to human readable time"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary"""
        total_time = time.time() - self.start_time
        return {
            "total_steps": self.total_steps,
            "completed_steps": self.current_step,
            "total_time_seconds": total_time,
            "total_time_formatted": self._format_time(total_time),
            "step_times": self.step_times,
            "average_step_time": sum(self.step_times) / len(self.step_times) if self.step_times else 0
        }


def print_section(title: str, char: str = "="):
    """Print a formatted section header"""
    print("\n" + char * 80)
    print(f"  {title}")
    print(char * 80 + "\n")


def run_command(cmd: str, cwd: Optional[Path] = None, description: str = "", 
                capture: bool = True, show_output: bool = False) -> Optional[subprocess.CompletedProcess]:
    """Run a shell command and handle errors"""
    if description:
        print(f"🔄 {description}...")
    
    try:
        if show_output:
            # Show real-time output for long-running commands
            process = subprocess.Popen(
                cmd,
                shell=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            output_lines = []
            for line in process.stdout:
                print(f"   {line.rstrip()}")
                output_lines.append(line)
            
            process.wait()
            
            if process.returncode == 0:
                print(f"✅ {description} completed")
                result = subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout=''.join(output_lines),
                    stderr=''
                )
                return result
            else:
                print(f"❌ Error: {description} failed")
                print(f"   Exit code: {process.returncode}")
                return None
        else:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                check=True,
                capture_output=capture,
                text=True
            )
            print(f"✅ {description} completed")
            return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description} failed")
        print(f"   Command: {cmd}")
        print(f"   Exit code: {e.returncode}")
        if capture and e.stderr:
            print(f"   Error output:\n{e.stderr[:500]}")
        return None


def check_model_exists(model_name: str, model_path: Path) -> bool:
    """Check if a trained model exists"""
    if not model_path.exists():
        print(f"❌ {model_name} not found!")
        print(f"   Expected location: {model_path}")
        return False
    
    print(f"✅ {model_name} found: {model_path}")
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"   Size: {size_mb:.1f} MB")
    return True


def run_indexing_pt() -> bool:
    """Run document indexing for SPLADE-PT-BR"""
    print_section("Step 1a: Document Indexing (SPLADE-PT-BR)", "-")
    
    cmd = "SPLADE_CONFIG_NAME=config_splade_pt python -m splade.index"
    result = run_command(
        cmd,
        cwd=SPLADE_DIR,
        description="Indexing documents with SPLADE-PT-BR"
    )
    
    if result:
        index_dir = RESULTS_DIR / "pt" / "index"
        if index_dir.exists():
            index_files = list(index_dir.glob("*"))
            total_size = sum(f.stat().st_size for f in index_files if f.is_file())
            print(f"   Index files: {len(index_files)}")
            print(f"   Total size: {total_size / (1024 * 1024):.1f} MB")
        return True
    return False


def run_indexing_en() -> bool:
    """Run document indexing for SPLADE-EN (original model)"""
    print_section("Step 1b: Document Indexing (SPLADE-EN)", "-")
    
    # Check if SPLADE-EN model is available
    en_model_path = SPLADE_DIR / "experiments" / "en" / "checkpoint" / "model" / "model.tar"
    
    if not en_model_path.exists():
        print("⚠️  SPLADE-EN model not found. Downloading from HuggingFace...")
        print("   This is optional for comparison purposes.")
        
        # Try to download SPLADE-EN model
        download_cmd = """
python -c "
from transformers import AutoModelForMaskedLM, AutoTokenizer
model = AutoModelForMaskedLM.from_pretrained('naver/splade-cocondenser-ensembledistil')
tokenizer = AutoTokenizer.from_pretrained('naver/splade-cocondenser-ensembledistil')
print('Model downloaded successfully')
"
"""
        result = run_command(
            download_cmd,
            cwd=PROJECT_ROOT,
            description="Downloading SPLADE-EN model"
        )
        
        if not result:
            print("⚠️  Could not download SPLADE-EN. Skipping EN indexing.")
            print("   Comparison will be limited to PT model only.")
            return False
    
    # Create EN config if needed
    en_config_path = SPLADE_DIR / "conf" / "config_splade_en.yaml"
    if not en_config_path.exists():
        print("⚠️  SPLADE-EN config not found. Skipping EN indexing.")
        print("   To enable EN comparison, create config_splade_en.yaml")
        return False
    
    cmd = "SPLADE_CONFIG_NAME=config_splade_en python -m splade.index"
    result = run_command(
        cmd,
        cwd=SPLADE_DIR,
        description="Indexing documents with SPLADE-EN"
    )
    
    if result:
        index_dir = RESULTS_DIR / "en" / "index"
        if index_dir.exists():
            index_files = list(index_dir.glob("*"))
            total_size = sum(f.stat().st_size for f in index_files if f.is_file())
            print(f"   Index files: {len(index_files)}")
            print(f"   Total size: {total_size / (1024 * 1024):.1f} MB")
        return True
    return False


def run_retrieval_pt() -> Optional[Dict[str, Any]]:
    """Run retrieval and evaluation for SPLADE-PT-BR"""
    print_section("Step 2a: Retrieval & Evaluation (SPLADE-PT-BR)", "-")
    
    cmd = "SPLADE_CONFIG_NAME=config_splade_pt python -m splade.retrieve"
    result = run_command(
        cmd,
        cwd=SPLADE_DIR,
        description="Retrieving with SPLADE-PT-BR"
    )
    
    if result:
        perf_file = RESULTS_DIR / "pt" / "out" / "perf_all_datasets.json"
        if perf_file.exists():
            with open(perf_file, 'r') as f:
                results = json.load(f)
            
            print("\n📊 SPLADE-PT-BR Results:")
            for dataset, metrics in results.items():
                print(f"\n   Dataset: {dataset}")
                for metric, value in metrics.items():
                    print(f"      {metric}: {value:.4f}")
            return results
    return None


def run_retrieval_en() -> Optional[Dict[str, Any]]:
    """Run retrieval and evaluation for SPLADE-EN"""
    print_section("Step 2b: Retrieval & Evaluation (SPLADE-EN)", "-")
    
    en_config_path = SPLADE_DIR / "conf" / "config_splade_en.yaml"
    if not en_config_path.exists():
        print("⚠️  SPLADE-EN config not found. Skipping EN retrieval.")
        return None
    
    cmd = "SPLADE_CONFIG_NAME=config_splade_en python -m splade.retrieve"
    result = run_command(
        cmd,
        cwd=SPLADE_DIR,
        description="Retrieving with SPLADE-EN"
    )
    
    if result:
        perf_file = RESULTS_DIR / "en" / "out" / "perf_all_datasets.json"
        if perf_file.exists():
            with open(perf_file, 'r') as f:
                results = json.load(f)
            
            print("\n📊 SPLADE-EN Results:")
            for dataset, metrics in results.items():
                print(f"\n   Dataset: {dataset}")
                for metric, value in metrics.items():
                    print(f"      {metric}: {value:.4f}")
            return results
    return None


def save_metrics_to_file(pt_results: Dict[str, Any], 
                        en_results: Optional[Dict[str, Any]],
                        output_dir: Path) -> Path:
    """Save detailed metrics to structured files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save PT-BR metrics
    pt_file = output_dir / f"metrics_pt_br_{timestamp}.json"
    with open(pt_file, 'w') as f:
        json.dump({
            "model": "SPLADE-PT-BR",
            "timestamp": datetime.now().isoformat(),
            "metrics": pt_results
        }, f, indent=2)
    print(f"✅ PT-BR metrics saved: {pt_file}")
    
    # Save EN metrics if available
    if en_results:
        en_file = output_dir / f"metrics_en_{timestamp}.json"
        with open(en_file, 'w') as f:
            json.dump({
                "model": "SPLADE-EN",
                "timestamp": datetime.now().isoformat(),
                "metrics": en_results
            }, f, indent=2)
        print(f"✅ EN metrics saved: {en_file}")
    
    return output_dir


def generate_comparison_report(pt_results: Dict[str, Any], 
                               en_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate detailed comparison report"""
    print_section("Generating Comparison Report", "-")
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "models": {
            "splade_pt_br": {
                "name": "SPLADE-PT-BR",
                "language": "Portuguese (Brazilian)",
                "base_model": "neuralmind/bert-base-portuguese-cased",
                "dataset": "mMARCO Portuguese",
                "results": pt_results
            }
        },
        "comparison": {}
    }
    
    if en_results:
        comparison["models"]["splade_en"] = {
            "name": "SPLADE-EN (Original)",
            "language": "English",
            "base_model": "naver/splade-cocondenser-ensembledistil",
            "dataset": "Same Portuguese dataset (cross-lingual)",
            "results": en_results
        }
        
        # Compare metrics
        print("\n📊 Detailed Comparison:")
        print("-" * 80)
        
        # Get all unique metrics
        all_metrics = set()
        for dataset_results in [pt_results, en_results]:
            for dataset, metrics in dataset_results.items():
                all_metrics.update(metrics.keys())
        
        for metric in sorted(all_metrics):
            comparison["comparison"][metric] = {}
            
            # Get PT value
            pt_value = None
            for dataset, metrics in pt_results.items():
                if metric in metrics:
                    pt_value = metrics[metric]
                    break
            
            # Get EN value
            en_value = None
            for dataset, metrics in en_results.items():
                if metric in metrics:
                    en_value = metrics[metric]
                    break
            
            if pt_value is not None and en_value is not None:
                diff = pt_value - en_value
                pct_diff = (diff / en_value * 100) if en_value != 0 else 0
                
                comparison["comparison"][metric] = {
                    "splade_pt_br": pt_value,
                    "splade_en": en_value,
                    "difference": diff,
                    "percent_difference": pct_diff
                }
                
                winner = "PT-BR" if pt_value > en_value else "EN" if en_value > pt_value else "TIE"
                symbol = "🏆" if winner == "PT-BR" else "📊" if winner == "TIE" else "📉"
                
                print(f"\n{symbol} {metric}:")
                print(f"   SPLADE-PT-BR: {pt_value:.4f}")
                print(f"   SPLADE-EN:    {en_value:.4f}")
                print(f"   Difference:   {diff:+.4f} ({pct_diff:+.2f}%)")
                print(f"   Winner:       {winner}")
    else:
        print("\n⚠️  No SPLADE-EN results available for comparison")
        print("   Report will contain PT-BR results only")
    
    # Save comparison report to multiple locations
    report_paths = []
    
    # Save to experiments directory
    report_path_1 = RESULTS_DIR / "comparison_report_detailed.json"
    report_path_1.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path_1, 'w') as f:
        json.dump(comparison, f, indent=2)
    report_paths.append(report_path_1)
    
    # Save to evaluation results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path_2 = EVALUATION_OUTPUT_DIR / f"comparison_report_{timestamp}.json"
    report_path_2.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path_2, 'w') as f:
        json.dump(comparison, f, indent=2)
    report_paths.append(report_path_2)
    
    print(f"\n✅ Comparison reports saved:")
    for path in report_paths:
        print(f"   • {path}")
    
    return comparison


def run_visualization(comparison_available: bool) -> bool:
    """Generate comparative visualizations"""
    print_section("Step 4: Generating Visualizations", "-")
    
    cmd = "python scripts/utils/visualize_results.py"
    result = run_command(
        cmd,
        cwd=PROJECT_ROOT,
        description="Generating visualizations"
    )
    
    if result:
        plots_dir = PROJECT_ROOT / "docs" / "images" / "plots"
        if plots_dir.exists():
            plots = list(plots_dir.glob("*.png"))
            print(f"\n   Generated {len(plots)} plots:")
            for plot in sorted(plots):
                size_kb = plot.stat().st_size / 1024
                print(f"      • {plot.name} ({size_kb:.1f} KB)")
        
        if comparison_available:
            print("\n   📊 Comparative plots include:")
            print("      • Side-by-side metric comparison")
            print("      • Performance difference charts")
            print("      • Sparsity comparison")
        
        return True
    return False


def save_execution_summary(progress: ProgressTracker, results: Dict[str, bool],
                          comparison_available: bool, output_dir: Path):
    """Save execution summary to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"execution_summary_{timestamp}.json"
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "progress": progress.get_summary(),
        "steps": results,
        "comparison_available": comparison_available,
        "success": all(results.values()),
        "output_locations": {
            "pt_br_metrics": str(RESULTS_DIR / "pt" / "out" / "perf_all_datasets.json"),
            "en_metrics": str(RESULTS_DIR / "en" / "out" / "perf_all_datasets.json") if comparison_available else None,
            "comparison_report": str(RESULTS_DIR / "comparison_report_detailed.json"),
            "plots": str(PROJECT_ROOT / "docs" / "images" / "plots"),
            "evaluation_results": str(output_dir)
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Execution summary saved: {summary_file}")
    return summary_file


def print_summary(progress: ProgressTracker, results: Dict[str, bool], 
                 comparison_available: bool):
    """Print evaluation summary"""
    print_section("Evaluation Summary")
    
    summary = progress.get_summary()
    print(f"⏱️  Total time: {summary['total_time_formatted']}")
    print(f"📊 Completed: {summary['completed_steps']}/{summary['total_steps']} steps")
    
    print("\n✅ Step Results:")
    for step, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {step}")
    
    if all(results.values()):
        print("\n🎉 Evaluation completed successfully!")
        
        print("\n📁 Results Location:")
        print(f"   • Evaluation Results: {EVALUATION_OUTPUT_DIR}/")
        print(f"   • PT-BR Metrics: {RESULTS_DIR}/pt/out/perf_all_datasets.json")
        if comparison_available:
            print(f"   • EN Metrics: {RESULTS_DIR}/en/out/perf_all_datasets.json")
        print(f"   • Comparison: {RESULTS_DIR}/comparison_report_detailed.json")
        print(f"   • Plots: {PROJECT_ROOT}/docs/images/plots/")
        
        print("\n📊 Key Findings:")
        if comparison_available:
            print("   ✓ Full comparison between SPLADE-PT-BR and SPLADE-EN available")
            print("   ✓ Both models evaluated on Portuguese dataset")
            print("   ✓ Performance differences quantified")
        else:
            print("   ✓ SPLADE-PT-BR evaluation complete")
            print("   ⚠ SPLADE-EN comparison not available (EN model not found)")
        
        print("\n📖 Next Steps:")
        print("   1. Review metrics in evaluation_results/ directory")
        print("   2. Check comparison_report_detailed.json for insights")
        print("   3. Analyze visualizations in docs/images/plots/")
        print("   4. Consider testing on additional datasets (see EVALUATION_PLAN.md)")
    else:
        print("\n⚠️  Some steps failed. Check errors above.")
        print("   Review logs and try running failed steps individually.")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation with model comparison for SPLADE-PT-BR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation with comparison
  python scripts/evaluation/run_evaluation_comparator.py
  
  # Skip already completed steps
  python scripts/evaluation/run_evaluation_comparator.py --skip-index-pt --skip-index-en
  
  # Evaluate PT model only (no comparison)
  python scripts/evaluation/run_evaluation_comparator.py --pt-only
        """
    )
    
    parser.add_argument("--skip-index-pt", action="store_true",
                       help="Skip indexing for SPLADE-PT-BR")
    parser.add_argument("--skip-index-en", action="store_true",
                       help="Skip indexing for SPLADE-EN")
    parser.add_argument("--skip-retrieve-pt", action="store_true",
                       help="Skip retrieval for SPLADE-PT-BR")
    parser.add_argument("--skip-retrieve-en", action="store_true",
                       help="Skip retrieval for SPLADE-EN")
    parser.add_argument("--skip-visualization", action="store_true",
                       help="Skip visualization generation")
    parser.add_argument("--pt-only", action="store_true",
                       help="Only evaluate SPLADE-PT-BR (no comparison)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("  SPLADE-PT-BR Evaluation & Comparison Pipeline")
    print("=" * 80)
    print(f"\n📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Project: {PROJECT_ROOT}")
    print(f"📁 Output: {EVALUATION_OUTPUT_DIR}/")
    
    if args.pt_only:
        print("🔍 Mode: PT-BR evaluation only (no comparison)")
    else:
        print("🔍 Mode: Full evaluation with EN comparison")
    
    # Calculate total steps
    total_steps = 1  # Prerequisites
    if not args.skip_index_pt:
        total_steps += 1
    if not args.pt_only and not args.skip_index_en:
        total_steps += 1
    if not args.skip_retrieve_pt:
        total_steps += 1
    if not args.pt_only and not args.skip_retrieve_en:
        total_steps += 1
    total_steps += 1  # Comparison report
    if not args.skip_visualization:
        total_steps += 1
    
    progress = ProgressTracker(total_steps)
    results = {}
    pt_results = None
    en_results = None
    
    # Check prerequisites
    progress.start_step("Prerequisites Check")
    pt_model_path = SPLADE_DIR / "experiments" / "pt" / "checkpoint" / "model_ckpt" / "model_final_checkpoint.tar"
    
    if not check_model_exists("SPLADE-PT-BR", pt_model_path):
        print("\n❌ SPLADE-PT-BR model not found. Please train it first:")
        print("   python scripts/training/train_splade_pt.py")
        sys.exit(1)
    progress.end_step(True)
    
    # Run SPLADE-PT-BR evaluation
    if not args.skip_index_pt:
        progress.start_step("PT-BR Document Indexing")
        results["PT-BR Indexing"] = run_indexing_pt()
        progress.end_step(results["PT-BR Indexing"])
        if not results["PT-BR Indexing"]:
            print("\n⚠️  PT-BR indexing failed. Cannot continue.")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping PT-BR indexing (--skip-index-pt)")
        results["PT-BR Indexing"] = True
    
    if not args.skip_retrieve_pt:
        progress.start_step("PT-BR Retrieval & Evaluation")
        pt_results = run_retrieval_pt()
        results["PT-BR Retrieval"] = pt_results is not None
        progress.end_step(results["PT-BR Retrieval"])
        if not results["PT-BR Retrieval"]:
            print("\n⚠️  PT-BR retrieval failed. Cannot continue.")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping PT-BR retrieval (--skip-retrieve-pt)")
        results["PT-BR Retrieval"] = True
        # Try to load existing results
        perf_file = RESULTS_DIR / "pt" / "out" / "perf_all_datasets.json"
        if perf_file.exists():
            with open(perf_file, 'r') as f:
                pt_results = json.load(f)
    
    # Run SPLADE-EN evaluation (if not PT-only mode)
    comparison_available = False
    if not args.pt_only:
        if not args.skip_index_en:
            progress.start_step("EN Document Indexing")
            results["EN Indexing"] = run_indexing_en()
            progress.end_step(results["EN Indexing"])
        else:
            print("\n⏭️  Skipping EN indexing (--skip-index-en)")
            results["EN Indexing"] = True
        
        if results.get("EN Indexing", False):
            if not args.skip_retrieve_en:
                progress.start_step("EN Retrieval & Evaluation")
                en_results = run_retrieval_en()
                results["EN Retrieval"] = en_results is not None
                progress.end_step(results["EN Retrieval"])
            else:
                print("\n⏭️  Skipping EN retrieval (--skip-retrieve-en)")
                results["EN Retrieval"] = True
                # Try to load existing results
                perf_file = RESULTS_DIR / "en" / "out" / "perf_all_datasets.json"
                if perf_file.exists():
                    with open(perf_file, 'r') as f:
                        en_results = json.load(f)
            
            comparison_available = en_results is not None
    
    # Save metrics to files
    if pt_results:
        progress.start_step("Saving Metrics & Generating Reports")
        save_metrics_to_file(pt_results, en_results, EVALUATION_OUTPUT_DIR)
        comparison = generate_comparison_report(pt_results, en_results)
        results["Comparison Report"] = True
        progress.end_step(True)
    else:
        print("\n⚠️  No PT-BR results available. Cannot generate comparison.")
        results["Comparison Report"] = False
    
    # Generate visualizations
    if not args.skip_visualization:
        progress.start_step("Generating Visualizations")
        results["Visualization"] = run_visualization(comparison_available)
        progress.end_step(results["Visualization"])
    else:
        print("\n⏭️  Skipping visualization (--skip-visualization)")
        results["Visualization"] = True
    
    # Save execution summary
    save_execution_summary(progress, results, comparison_available, EVALUATION_OUTPUT_DIR)
    
    # Print summary
    print_summary(progress, results, comparison_available)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

