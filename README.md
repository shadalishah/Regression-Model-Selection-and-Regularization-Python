# 🎯 Regression Modeling & Feature Selection — Ridge, Lasso, PCR, PLS & Subset Selection

> **Skills Demonstrated:** Lasso Regression · Ridge Regression · Best Subset Selection · Forward/Backward Stepwise Selection · Principal Components Regression (PCR) · Partial Least Squares (PLS) · Cross-Validation · Regularization · Bias-Variance Tradeoff · Python · Scikit-learn · Statsmodels

[![Author](https://img.shields.io/badge/Author-Shad%20Ali%20Shah-blue)](https://github.com/shadalishah)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/shad-ali-shah-6439ab339/)
[![Language](https://img.shields.io/badge/Language-Python-3776AB?logo=python)](https://www.python.org/)
[![Topic](https://img.shields.io/badge/Topic-Linear%20Model%20Selection%20%26%20Regularization-green)]()

---

## 🎯 Project Overview

This project implements and benchmarks **five advanced regression and feature selection methods** — from penalized regression to dimensionality reduction — applied to real-world prediction problems in higher education and urban economics.

> *"In production ML, the question is never just 'which model?' — it's 'which features?' and 'how do we avoid overfitting?' This project demonstrates exactly those decisions: feature selection, regularization tuning, and honest out-of-sample evaluation."*

Four exercises covered across three datasets:

| Exercise | Dataset | Business Question |
|----------|---------|-----------------|
| Ex. 8 | Simulated | Can selection methods recover true predictors from noisy data? |
| Ex. 9 | College (777 universities) | Best model for predicting university applications? |
| Ex. 10 | Simulated (p=20, n=1000) | When does adding features hurt prediction? |
| Ex. 11 | Boston Housing (506 suburbs) | Which model best predicts crime rates? |

---

## 📁 Datasets Used

| Dataset | Source | Size | Target Variable |
|---------|--------|------|----------------|
| **Simulated (Ex. 8)** | `numpy.random` | 100 obs, 10 features | Polynomial signal + noise |
| **College** | U.S. News & World Report 1995 (Real) | 777 universities, 18 features | Apps — applications received |
| **Simulated (Ex. 10)** | `numpy.random` | 1,000 obs, 20 features | 10 truly non-zero coefficients |
| **Boston** | U.S. Census Bureau (Real) | 506 suburbs, 13 features | crim — per capita crime rate |

---

## 🔧 Techniques & Tools Applied

| Technique | Library | Purpose |
|-----------|---------|---------|
| Best Subset Selection | `l0bnb` | Exhaustive optimal feature subset |
| Forward Stepwise Selection | `ISLP` | Greedy forward feature addition |
| Backward Stepwise Selection | `ISLP` | Greedy backward feature removal |
| **Ridge Regression (L2)** | `sklearn.linear_model.Ridge` | Coefficient shrinkage, no sparsity |
| **Lasso Regression (L1)** | `sklearn.linear_model.Lasso` | Automatic variable selection |
| PCR | `sklearn.decomposition.PCA` + OLS | Dimensionality reduction regression |
| PLS | `sklearn.cross_decomposition.PLSRegression` | Supervised dimensionality reduction |
| K-Fold Cross-Validation | `sklearn.model_selection.KFold` | Objective λ / component selection |
| Cp, BIC, Adjusted R² | `ISLP` | Information criteria model selection |

**Libraries:** `numpy` · `pandas` · `scikit-learn` · `statsmodels` · `matplotlib` · `ISLP` · `l0bnb`

---

## 📊 Key Results

### Exercise 8 — Recovering True Predictors from Simulated Noisy Data

**True model:** Y = β₀ + β₁X + β₂X² + β₃X³ + ε

#### Forward & Backward Stepwise Selection:

| Criterion | Selected Variables | Correct? |
|-----------|-------------------|---------|
| **Cp** | **X, X², X³** | ✅ **Exact match** |
| BIC | X, X², X³ | ✅ Exact match |
| Adjusted R² | X, X², X³ | ✅ Exact match |

#### Lasso Regression (CV-tuned λ):

| Variables Selected | Count | Notes |
|-------------------|-------|-------|
| X, X², X³ | Core | ✅ Correctly identified |
| X⁴, X⁹ | Extra | ⚠️ Minor over-selection |

> **Finding:** Stepwise selection perfectly recovered the true 3-variable model. Lasso selected a slightly broader set (including X⁴ and X⁹) — reflecting the regularization vs. greedy selection tradeoff.

**Sparse true model test (β₇ only):**
- Forward stepwise selection → **recovered X⁷ precisely** ✅
- Demonstrates method's power in truly sparse signal settings

---

### Exercise 9 — College Applications Prediction (5-Model Benchmark)

**Setup:** 777 universities, 17 predictors → target: `Apps`
**Split:** Random 50/50 train/test

#### Full Model Comparison:

| Model | Method | Components/Features | Test RMSE | vs OLS |
|-------|--------|-------------------|----------|--------|
| **OLS** | Standard least squares | All 17 | **Lowest** | ✅ Baseline best |
| Ridge | L2, CV-tuned λ | All 17 (shrunk) | Close to OLS | ~Same |
| **Lasso** | L1, CV-tuned λ | **15 non-zero** | Comparable | ~Same |
| PCR | PCA + OLS | **17 components** (CV) | Comparable | ~Same |
| PLS | Supervised reduction | **12 components** (CV) | Comparable | ~Same |

**Model spread:** Maximum RMSE variation across all 5 methods = **4.36%**

> **Key Finding:** OLS linear regression outperformed all regularized models — indicating this dataset is **not strongly affected by overfitting**. All five methods produced nearly identical predictions (within ±4.36%) — demonstrating that for well-conditioned data, method choice matters less than feature quality.
>
> Average prediction error across all models: **±~57 applications per university** — impressively accurate relative to the dataset mean of ~3,002 applications.

---

### Exercise 10 — Bias-Variance Tradeoff in Action (Simulated p=20)

**Setup:** n=1,000 obs, p=20 features, only **10 truly non-zero** coefficients

#### Best Subset Selection Results:

| Model Size | Training MSE | Test MSE | Coefficient Error |
|-----------|-------------|---------|------------------|
| 1 feature | High | High | High |
| **3 features** | Medium | **Lowest** ✅ | Medium |
| **6 features** | Lower | Higher | **Lowest** ✅ |
| 10 features | Lower | Higher | — |
| 20 features | **Lowest** | **Highest** | High |

> **Critical Finding:** Training MSE decreased **continuously** as predictors were added — confirming training error alone is a **dangerously misleading** guide to model quality.
>
> Three distinct optima emerge: best **test MSE at 3 features**, best **coefficient recovery at 6 features**, and worst performance at 20 features (all noise included). This cleanly demonstrates the **bias-variance tradeoff** — a finding directly applicable to any production feature selection workflow.

---

### Exercise 11 — Boston Crime Rate Prediction (4-Method Comparison)

**Setup:** 506 suburbs, 12 predictors → target: `crim` (per capita crime)
**Train/Test Split:** CV-based evaluation

#### Model Comparison:

| Model | Features Used | Test MSE | Interpretable? |
|-------|--------------|---------|---------------|
| **Best Subset (2 features)** | 2 selected | **Lowest** | ✅ Most interpretable |
| Lasso (CV-tuned) | Sparse subset | Comparable | ✅ |
| Ridge (CV-tuned) | All 12 (shrunk) | Comparable | ⚠️ Less interpretable |
| PCR | CV components | Comparable | ❌ Least interpretable |

**Best Subset Selected Features:**
- **`rad`** (highway accessibility) — strongest crime predictor
- **`medv`** (median home value) — negative predictor

> **Recommendation:** Best subset selection with **2 predictors** — balancing accuracy and simplicity, essential for real-world deployment and stakeholder communication. Ridge achieves similar accuracy but sacrifices interpretability.
>
> **Key Insight:** Highway accessibility (`rad`) alone is the strongest predictor of crime in Boston suburbs. Any urban crime model that omits this variable is misspecified.

---

## 📊 Overall Method Comparison Summary

| Method | Strengths | Best Used When |
|--------|-----------|---------------|
| **Best Subset** | Optimal feature combination | Low p, need exact selection |
| **Forward/Backward Stepwise** | Scalable greedy selection | Medium p, computational limits |
| **Lasso (L1)** | Automatic sparsity, sparse output | High p, need automatic selection |
| **Ridge (L2)** | Stable with correlated features | Multicollinearity present |
| **PCR** | Handles collinearity | Highly correlated predictors |
| **PLS** | Supervised dimensionality reduction | When Y-X correlation matters |

---

## 💡 Business Insights

1. **Feature Selection Before Modeling:** Exercise 10 proved that adding all 20 features worsened test MSE by 2× vs the 3-feature optimal model. In production, automatic feature selection (Lasso or stepwise) should precede any deployment decision.

2. **Regularization Isn't Always Needed:** For the College dataset, OLS outperformed Ridge and Lasso — the data was not overfit-prone. Applying regularization blindly adds complexity without benefit. Always verify with CV.

3. **Training Error is Misleading:** Training MSE decreased monotonically with model size even when test MSE was rising. This is why cross-validation is mandatory — it's the only reliable guide to production performance.

4. **Lasso for Interpretability:** In the Boston crime analysis, Lasso automatically eliminated irrelevant predictors — producing an interpretable sparse model. For regulatory compliance and stakeholder reporting, sparse models are often preferred over black-box dense ones.

5. **2 Features Can Beat 12:** Best subset selection with just `rad` and `medv` matched full-model Ridge and PCR performance on Boston crime prediction. Simpler models are easier to maintain, explain, and monitor in production.

---

## 🗂️ File Structure

```
Chapter_6_Applied_Exercise_Solutions/
│
├── Chapter_6.ipynb          ← Main analysis notebook (all exercises)
├── chapter_6.html           ← Rendered HTML version (easy browser viewing)
├── chapter_6.qmd            ← Quarto source file
└── README.md                ← This file
```

---

## ▶️ How to Run

```bash
# Install dependencies
pip install ISLP scikit-learn statsmodels pandas numpy matplotlib l0bnb jupyter

# Launch notebook
jupyter notebook Chapter_6.ipynb
```

> **Note:** Exercise 10 uses the `l0bnb` package for best subset selection. Install with `pip install l0bnb` if needed.

---

## 📚 Reference

James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023).
*An Introduction to Statistical Learning with Applications in Python.* Springer.
Chapter 6: Linear Model Selection and Regularization — Applied Exercises 8–11.

---

## 🙏 Acknowledgements

Special thanks to **Karim Aboussel Ham** whose repository
[ISLP-applied-solutions](https://github.com/KarimABOUSSELHAM/ISLP-applied-solutions)
provided useful guidance and reference during the completion of this project.

---

## 👤 About the Author

**Shad Ali Shah**
🎓 MPhil Economics Student — School of Economics, Quaid-i-Azam University, Islamabad
💡 Passionate about the intersection of **Economics**, **Data Science**, and **Machine Learning**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/shad-ali-shah-6439ab339/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/shadalishah)

---

*Part of the [ML Portfolio](../README.md) by Shad Ali Shah*
