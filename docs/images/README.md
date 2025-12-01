# 📊 Visualizations

High-quality plots for training analysis and documentation.

## Available Plots

- `training_loss_enhanced.png` - Training convergence analysis
- `sparsity_analysis_dashboard.png` - Sparsity and efficiency metrics  
- `project_summary_dashboard.png` - Project overview
- `metrics_comparison_enhanced.png` - Performance comparison (after evaluation)

## Generate Plots

```bash
python scripts/utils/visualize_results.py
```

For metrics comparison, run evaluation first:
```bash
cd splade && python -m splade.index +config=config_splade_pt
cd splade && python -m splade.retrieve +config=config_splade_pt
python scripts/utils/compare_models.py
python scripts/utils/visualize_results.py
```
