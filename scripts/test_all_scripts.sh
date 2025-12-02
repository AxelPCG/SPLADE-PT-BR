#!/bin/bash
# Test scripts functionality

cd "/home/user/Projects/SPLADE-PT-BR"

pass=0
fail=0

check() {
    if eval "$2" >/dev/null 2>&1; then
        echo "✓ $1"
        ((pass++))
    else
        echo "✗ $1"
        ((fail++))
    fi
}

echo "Python syntax:"
check "main.py" "python3 -m py_compile main.py"
check "train_splade_pt.py" "python3 -m py_compile scripts/training/train_splade_pt.py"
check "compare_models.py" "python3 -m py_compile scripts/utils/compare_models.py"
check "visualize_results.py" "python3 -m py_compile scripts/utils/visualize_results.py"
check "run_evaluation_comparator.py" "python3 -m py_compile scripts/evaluation/run_evaluation_comparator.py"

echo ""
echo "Script help:"
check "train --help" "python3 scripts/training/train_splade_pt.py --help"
check "eval --help" "python3 scripts/evaluation/run_evaluation_comparator.py --help"

echo ""
echo "Dependencies:"
check "matplotlib" "python3 -c 'import matplotlib'"
check "numpy" "python3 -c 'import numpy'"
check "scipy" "python3 -c 'import scipy'"

echo ""
echo "Files:"
check "README.md" "[ -f README.md ]"
check "docs/USAGE.md" "[ -f docs/USAGE.md ]"
check "docs/MODEL_CARD.md" "[ -f docs/MODEL_CARD.md ]"
check "pyproject.toml" "[ -f pyproject.toml ]"

echo ""
echo "Directories:"
check "docs/" "[ -d docs ]"
check "scripts/training/" "[ -d scripts/training ]"
check "scripts/utils/" "[ -d scripts/utils ]"
check "scripts/evaluation/" "[ -d scripts/evaluation ]"
check "notebooks/" "[ -d notebooks ]"
check "docs/images/plots/" "[ -d docs/images/plots ]"

echo ""
echo "$pass passed, $fail failed"
[ $fail -eq 0 ] && exit 0 || exit 1
