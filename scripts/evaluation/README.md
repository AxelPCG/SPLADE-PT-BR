# 📊 Evaluation

## Quick Start

### PT-BR Only (Current)

```bash
python scripts/evaluation/run_evaluation_comparator.py --pt-only
```

### Full Comparison (PT-BR vs EN)

```bash
# This will download SPLADE-EN model and compare both on Portuguese dataset
python scripts/evaluation/run_evaluation_comparator.py
```

**What this does:**
1. Downloads SPLADE-EN model from HuggingFace (`naver/splade-cocondenser-ensembledistil`)
2. Indexes mRobust documents with both models
3. Evaluates both models on same 250 Portuguese queries
4. Generates detailed comparison metrics

## Results

After running, check:
- `evaluation_results/metrics_pt_br_*.json` - PT-BR metrics
- `evaluation_results/metrics_en_*.json` - EN metrics (if comparison ran)
- `evaluation_results/comparison_report_*.json` - Side-by-side comparison
- `evaluation_results/execution_summary_*.json` - Timing and execution details

## Metrics Collected

- **MRR@10** (Mean Reciprocal Rank)
- **Recall@100** 
- **Recall@1000**
- **NDCG@10** (Normalized Discounted Cumulative Gain)
- **MAP** (Mean Average Precision)

## Skip Completed Steps

```bash
# Skip indexing if already done
python scripts/evaluation/run_evaluation_comparator.py --skip-index-pt --skip-index-en

# Skip retrieval if already done
python scripts/evaluation/run_evaluation_comparator.py --skip-retrieve-pt --skip-retrieve-en
```

