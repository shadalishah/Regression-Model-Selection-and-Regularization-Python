# Regression Modelling & Feature Selection in Python
### Chapter 6 Applied Exercises — *An Introduction to Statistical Learning (ISLR2)*

[![Author](https://img.shields.io/badge/Author-Shad%20Ali%20Shah-blue)](https://github.com/shadalishah)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/shad-ali-shah-6439ab339/)
[![Language](https://img.shields.io/badge/Language-Python-3776AB?logo=python)](https://www.python.org/)
[![Topic](https://img.shields.io/badge/Topic-Linear%20Model%20Selection%20%26%20Regularization-green)]()

---

## 🔍 What This Project Is About

This project demonstrates advanced **predictive modelling and model selection techniques** using Python, applied to both real-world and simulated datasets. The work covers a core set of skills used daily by data scientists and quantitative analysts: building regression models, selecting the most important variables, and tuning models to avoid overfitting.

The exercises are drawn from **Chapter 6: Linear Model Selection and Regularization** of *An Introduction to Statistical Learning with Applications in Python* (ISLR2) — a standard reference text used in university programs and data science training worldwide.

> **In simple terms:** I built multiple types of prediction models, compared their accuracy, and identified which approach works best — skills directly applicable to forecasting, risk modelling, and data-driven decision-making.

---

## 💼 Skills Demonstrated

| Skill | What I Did |
|---|---|
| **Regularization (Lasso & Ridge)** | Built shrinkage models to reduce overfitting and improve out-of-sample accuracy |
| **Cross-Validation** | Used k-fold CV to objectively select the best tuning parameter (λ) |
| **Stepwise Variable Selection** | Applied forward & backward selection to identify the most predictive features |
| **Best Subset Selection** | Exhaustively compared all possible feature combinations to find the optimal model |
| **Dimensionality Reduction (PCR & PLS)** | Reduced high-dimensional data into components for more stable regression |
| **Model Comparison** | Benchmarked 5+ models using Test MSE to recommend the best-performing approach |
| **Data Simulation** | Generated controlled datasets to validate model behaviour under known conditions |
| **Python & Libraries** | `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `statsmodels`, `ISLP` |

---

## 📂 Exercises Solved

---

### 🧪 Exercise 8 — *How well can we recover a true signal from noisy data?*

**What I did:** Generated simulated data from a known polynomial model and tested whether different selection methods could correctly identify the true predictors.

**Methods applied:** Forward Stepwise Selection · Backward Stepwise Selection · Lasso Regression

**Key findings:**
- Both forward and backward stepwise selection correctly identified **X, X², and X³** as the relevant predictors using the Cp criterion — matching the true model.
- The Lasso, using cross-validated λ, selected a slightly broader set of variables (including X⁴ and X⁹), reflecting the greedy vs. regularization tradeoff.
- When the true model was simplified to a single dominant term (X⁷), forward stepwise selection recovered the correct model precisely — demonstrating the method's practical power in sparse settings.

---

### 🎓 Exercise 9 — *Can we predict how many applications a university will receive?*

**What I did:** Used the **College dataset** (777 US universities) to build and compare five different regression models for predicting the number of applications received.

**Methods applied:** OLS Linear Regression · Ridge Regression · Lasso Regression · Principal Components Regression (PCR) · Partial Least Squares (PLS)

**Model comparison summary:**

| Model | Approach | Result |
|---|---|---|
| OLS | Standard least squares | ✅ Best test MSE |
| Ridge | L2 regularization (CV-tuned λ) | Close to OLS |
| Lasso | L1 regularization — 15 non-zero coefficients | Comparable |
| PCR | 17 components selected by CV | Comparable |
| PLS | 12 components selected by CV | Comparable |

**Key findings:**
- OLS linear regression outperformed all regularized models on this dataset, suggesting the data is not strongly affected by overfitting.
- All five models produced very similar RMSE values — maximum variation of only **4.36%** — indicating robust, consistent prediction regardless of method.
- The average prediction error across models was approximately **±57 applications** per university, which is impressively accurate relative to the dataset mean.

---

### 🔢 Exercise 10 — *When does adding more features hurt your model?*

**What I did:** Generated a dataset with 20 features and 1,000 observations (with only 10 truly non-zero coefficients) to study the bias-variance tradeoff in action.

**Methods applied:** Best Subset Selection · Training vs. Test MSE comparison

**Key findings:**
- Training MSE decreased continuously as more predictors were added — confirming that training error alone is a **misleading guide** to model quality.
- Test MSE was minimized at an intermediate model size — demonstrating the classic overfitting curve.
- The model with the lowest coefficient error (6 predictors) was not the same as the model with the lowest test MSE (3 predictors) — proving that fitting coefficients well does not guarantee the best predictive performance.

---

### 🏙️ Exercise 11 — *What predicts crime rates in Boston neighbourhoods?*

**What I did:** Applied four different regression methods to the **Boston Housing dataset** to predict per capita crime rate and recommend the best model.

**Methods applied:** Best Subset Selection · Lasso Regression · Ridge Regression · PCR

**Key findings:**
- Best subset selection identified a **2-feature model** as the most accurate and interpretable.
- Ridge regression on the full model achieved comparable test MSE, but at the cost of interpretability.
- The recommended final model uses **best subset selection with 2 predictors** — balancing accuracy and simplicity, which is essential for real-world deployment and stakeholder communication.

---

## 📁 Repository Structure

```
📦 ISLR2-Chapter6-LinearModelSelection/
│
├── 📓 Chapter_6.ipynb     # Full Python notebook with code + explanations
├── 🌐 chapter_6.html      # Web-viewable version of the notebook
├── 📄 chapter_6.qmd       # Source file (Quarto format)
└── 📋 README.md           # This file
```

---

## ▶️ How to Run This Project

**Step 1 — Install Python dependencies:**
```bash
pip install numpy pandas matplotlib scikit-learn statsmodels islp jupyter
```

**Step 2 — Clone the repository:**
```bash
git clone https://github.com/shadalishah/ISLR2-Chapter6-LinearModelSelection.git
cd ISLR2-Chapter6-LinearModelSelection
```

**Step 3 — Open the notebook:**
```bash
jupyter notebook Chapter_6.ipynb
```

> **Note:** Exercise 10 uses the `l0bnb` package for best subset selection. Install it with `pip install l0bnb` if needed.

---

## 🙏 Acknowledgements

Special thanks to **[Karim Aboussel Ham](https://github.com/KarimABOUSSELHAM)** whose repository [ISLP-applied-solutions](https://github.com/KarimABOUSSELHAM/ISLP-applied-solutions) provided helpful code examples and guidance during the completion of these exercises.

---

## 👤 About the Author

**Shad Ali Shah**
MPhil Economics Student — School of Economics, Quaid-i-Azam University, Islamabad
Passionate about the intersection of **Economics, Data Science, and Machine Learning**

🔗 [LinkedIn](https://www.linkedin.com/in/shad-ali-shah-6439ab339/) &nbsp;|&nbsp; 🐙 [GitHub](https://github.com/shadalishah)

---

## 📚 Reference

> James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023).
> *An Introduction to Statistical Learning with Applications in Python*. Springer.
> [https://www.statlearning.com](https://www.statlearning.com)

---

