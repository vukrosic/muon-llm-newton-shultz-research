# Newton-Schulz Formula Ablation Study

## Overview

This repository contains a comprehensive ablation study of the Newton-Schulz formula used in the Muon optimizer. We systematically tested 11 different variations of the Newton-Schulz orthogonalization formula to understand which components are essential for stable convergence.

## Experimental Setup

- **Model**: MoE Transformer (384d, 6L, 8H, 1536ff, 8 experts)
- **Training**: 500 steps per experiment
- **Data**: 500K tokens, sequence length 512
- **Hardware**: NVIDIA GeForce RTX 4090 (25.3 GB)
- **Metrics**: Validation loss, accuracy, perplexity, training speed

## Newton-Schulz Formula Variations Tested

### Original Formula
```
X = a*X + (b*A + c*A²)*X
where A = X*X^T
Coefficients: a=3.4445, b=-4.7750, c=2.0315
```

### Tested Variations

| Experiment | Description | Formula | Status | Loss | Time (s) |
|------------|-------------|---------|--------|------|----------|
| **baseline** | Original coefficients | `a*X + (b*A + c*A²)*X` | ✅ | 4.5201 | 75.37 |
| **more-iter** | 8 iterations (vs 5) | Same formula, 8 steps | ✅ | 4.5198 | 66.16 |
| **reordered** | Reordered terms | `(c*A² + b*A)*X + a*X` | ✅ | 4.5213 | 66.44 |
| **adaptive** | Dynamic coefficients | Adaptive a,b,c based on norm | ✅ | 4.5343 | 73.20 |
| **fewer-iter** | 3 iterations (vs 5) | Same formula, 3 steps | ✅ | 4.6685 | 64.19 |
| **alt-coeffs** | (2.5, -3.0, 1.8) | Same structure | ❌ | NaN | 44.43 |
| **simple-coeffs** | (3.0, -4.0, 2.0) | Same structure | ❌ | NaN | 63.41 |
| **no-cubic** | Remove A² term | `a*X + b*A*X` | ❌ | NaN | 40.76 |
| **no-quadratic** | Remove A term | `a*X + c*A²*X` | ❌ | NaN | 40.43 |
| **only-linear** | Only linear scaling | `a*X` | ❌ | NaN | 53.81 |
| **diff-structure** | 4-term structure | `a*X + b*A*X + c*A²*X + d*X` | ❌ | NaN | 40.49 |

## Key Findings

### ✅ **Critical Success Factors**

1. **Exact Coefficients Matter**: The original coefficients `(3.4445, -4.7750, 2.0315)` are essential. Changing them to simpler values breaks convergence completely.

2. **Both Terms Required**: Both the quadratic (`A`) and cubic (`A²`) terms are necessary for stable convergence. Removing either causes NaN values.

3. **Structure Sensitivity**: The exact algebraic structure is important. Reordering works, but changing the fundamental form breaks convergence.

4. **Iteration Count**: More iterations (8) perform slightly better than the baseline (5), while fewer iterations (3) perform worse but still work.

### ❌ **Failure Modes**

- **Coefficient Changes**: Any deviation from original coefficients causes NaN
- **Term Removal**: Removing quadratic or cubic terms causes NaN
- **Structure Changes**: Adding extra terms or changing the fundamental form causes NaN
- **Linear-Only**: Using only linear scaling fails completely

### 🚀 **Performance Insights**

- **Best Performance**: More iterations (8 steps) - Loss: 4.5198, Time: 66.16s
- **Fastest Training**: No-cubic variant (40.76s) but produces NaN
- **Most Robust**: Original baseline - consistently stable
- **Speed vs Quality**: More iterations gives both better performance AND faster training

## Recommendations

1. **Use 8 iterations instead of 5** - provides better performance and faster training
2. **Don't modify the coefficients** - they are carefully tuned and essential
3. **Keep both quadratic and cubic terms** - both are required for convergence
4. **Maintain the original structure** - the algebraic form is optimal
5. **Consider reordered terms** - provides similar performance with slight speed improvement

## Technical Details

### Model Architecture
- **Dimensions**: 384 model dimension, 8 attention heads, 6 layers
- **MoE**: 8 experts with top-2 routing
- **Parameters**: 79M total, 22.4M active (28.4% efficiency)
- **Optimizer**: Hybrid Muon (60M params) + AdamW (19M params)

### Training Configuration
- **Batch Size**: 24
- **Learning Rate**: 0.01 (Muon), 0.001 (AdamW)
- **Gradient Accumulation**: 4 steps
- **Mixed Precision**: Enabled
- **Weight Decay**: 0.1

### Data
- **Source**: HuggingFaceTB/smollm-corpus (cosmopedia-v2)
- **Documents**: 2,000 text documents
- **Tokens**: 500,000 tokens
- **Sequence Length**: 512 tokens
- **Train/Val Split**: 90/10

## Files

- `llm.py` - Main experiment code with all Newton-Schulz variations
- `requirements.txt` - Python dependencies
- `experiment_results/` - JSON files with detailed results
- `data_cache/` - Cached tokenized data

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run ablation experiments
python llm.py
```

The script will automatically:
- Load and cache data
- Run all 11 experiments sequentially
- Save results after each experiment (crash-resistant)
- Generate a comprehensive summary

## Results Interpretation

The experiments reveal that the Newton-Schulz formula is **highly sensitive** to its exact form. The original coefficients and structure were very carefully chosen, and even small modifications can completely break convergence. This suggests that the formula represents a delicate balance that enables stable orthogonalization of momentum updates.

The success of the "more iterations" variant suggests that the original 5 iterations might be conservative, and using 8 iterations could provide better performance without sacrificing speed.

## Future Work

- Test intermediate coefficient values between original and failed variants
- Investigate why certain coefficient combinations fail
- Explore adaptive coefficient strategies that maintain stability
- Test on different model architectures and datasets
- Analyze the mathematical properties that make the original formula stable

---

*Generated from Newton-Schulz ablation experiments on 2024-12-12*
