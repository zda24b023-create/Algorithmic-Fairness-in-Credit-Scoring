# P26 — Algorithmic Fairness in Credit Scoring
### Does Model Compression Amplify or Reduce Demographic Bias?

**ML 2026 Course Project · IIT Madras Zanzibar**




## 👥 Authors

| Name | Roll Number | Dataset |
|---|---|---|
| Raudhat Mwinyi | zda24b010 | UCI German Credit |
| Ibrahim Silima Mnemba | zda24b023 | UCI Adult (Income) |

---

## 📌 Project Overview

This project investigates whether compressing a credit-scoring AI model — through **pruning** and **quantization** — makes it more or less fair toward demographic groups such as women, older people, and racial minorities.

We train classical ML models and MLP neural networks on two real-world credit datasets, compress the MLP using multiple techniques, and measure fairness before and after compression using Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD).

**Central Finding:** Pruning consistently *reduces* demographic bias in credit scoring models (opposite of what prior work found for image models), while quantization preserves bias at near-baseline levels.

---

## 🗂️ Repository Structure

```
P26-Algorithmic-Fairness-Credit-Scoring/
│
├── D1/
│   ├── D1_German_Credit.ipynb       # EDA + RF + MLP on German Credit
│   ├── D1_Adult.ipynb               # EDA + RF + MLP on UCI Adult
│   └── D1_Combined_P26.ipynb        # Combined notebook (both datasets)
│
├── D2/
│   ├── D2_German_Credit_FINAL.ipynb # All ML + PCA + Compression + Fairness (German)
│   ├── D2_Adult_FINAL.ipynb         # All ML + PCA + Compression + Fairness (Adult)
│   └── D2_Report_P26.pdf            # Preliminary report (NeurIPS format)
│
├── D3/
│   ├── D3_German_Credit.ipynb       # Improvement + comparison with studies (German)
│   └── D3_Adult.ipynb               # Improvement + comparison with studies (Adult)
│
├── reports/
│   ├── D1_Report_P26_Final.tex      # D1 LaTeX report
│   ├── D2_Report_P26_Fixed.tex      # D2 LaTeX report
│   └── D2_Report_P26_Final.pdf      # D2 compiled PDF
│
└── README.md
```

---

## 📊 Datasets

| Dataset | Samples | Features | Task | Source |
|---|---|---|---|---|
| UCI German Credit | 1,000 | 20 | Binary (good/bad credit) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| UCI Adult (Income) | 48,842 | 14 | Binary (>$50K income) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/2/adult) |

**Fairness attributes:**
- German Credit: Gender (Female=310, Male=690) and Age (Young<35 vs Senior≥35)
- UCI Adult: Sex (Female vs Male) and Race (White vs Non-White)

**Bias found in raw data (before any model is trained):**
- German Credit: 7.5% gender bias gap (Female bad credit 35.2% vs Male 27.7%)
- UCI Adult: 19.5% gender gap + 10.1% racial income gap

---

## 🔬 Methods

### Classical ML Models
All trained with 5 seeds `{42, 0, 1, 2, 3}`, 80/20 stratified split, results as mean ± std.

| Model | Library |
|---|---|
| Random Forest | scikit-learn |
| SVM (RBF kernel) | scikit-learn |
| KNN (k=5) | scikit-learn |
| Logistic Regression | scikit-learn |
| Decision Tree | scikit-learn |

### MLP Architecture
```
Input(d) → Linear(d, 64) → ReLU → Dropout(0.3)
         → Linear(64, 32) → ReLU → Dropout(0.3)
         → Linear(32, 1)
```
- Loss: BCEWithLogitsLoss with pos_weight (handles class imbalance)
- Optimizer: Adam, lr=0.001, weight_decay=1e-4, batch_size=32
- German Credit: fixed 50 epochs
- UCI Adult: early stopping (max 200 epochs, patience=10)

### Compression Techniques

| Technique | Method | Detail |
|---|---|---|
| Pruning | Magnitude-based L1 unstructured | 30%, 50%, 70% sparsity |
| Quantization | Dynamic int8 | 4× size reduction |
| Coreset Selection | K-Center Greedy | 50% of training data |
| Dimensionality Reduction | PCA | 90% variance threshold |

### Fairness Metrics
- **DPD (Demographic Parity Difference):** difference in positive prediction rates between groups
- **EOD (Equalized Odds Difference):** difference in true positive rates between groups
- Lower = fairer. Measured on fixed seed=42 test set.

---

## 📈 Key Results

### D1 — Baseline

| Dataset | Best Model | Accuracy | AUC |
|---|---|---|---|
| German Credit | Random Forest | 0.783 ± 0.018 | 0.801 |
| UCI Adult | MLP | 0.854 ± 0.002 | 0.910 |

### D2 — Compression & Fairness

**German Credit — Fairness (most important table):**

| Scenario | Gender DPD | Gender EOD | Age DPD | Age EOD |
|---|---|---|---|---|
| Base MLP | 0.1033 | 0.0544 | 0.2586 | 0.2064 |
| Pruned 30% | 0.0697 | 0.0276 | 0.1559 | 0.0921 |
| Pruned 50% | 0.0956 | 0.0697 | 0.1258 | 0.0354 |
| **Pruned 70%** | **0.0048 ↓95%** | **0.0059** | **0.0607 ↓77%** | **0.0049** |
| Quantized (int8) | 0.1105 | 0.0645 | 0.2692 | 0.2196 |

**UCI Adult — Fairness:**

| Scenario | Gender DPD | Gender EOD | Race DPD | Race EOD |
|---|---|---|---|---|
| Base MLP | 0.3250 | 0.1654 | 0.1637 | 0.0900 |
| **Pruned 70%** | **0.0819 ↓75%** | **0.0778** | **0.0835 ↓49%** | 0.1652 |
| Quantized (int8) | 0.3163 | 0.1382 | 0.1598 | 0.0808 |

> **Key finding:** Pruning reduces bias by up to 95% — directly contradicting Hooker et al. (ICLR 2020) who found compression amplifies bias in image models. For credit scoring tabular data, the effect is opposite.

---

## 🚀 How to Run

### Option 1 — Kaggle (recommended)
1. Upload any notebook to [Kaggle](https://kaggle.com)
2. Enable GPU: Settings → Accelerator → GPU
3. Run All cells

### Option 2 — Local
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/P26-Algorithmic-Fairness-Credit-Scoring.git
cd P26-Algorithmic-Fairness-Credit-Scoring

# Install dependencies
pip install torch scikit-learn pandas numpy matplotlib seaborn aif360

# Run a notebook
jupyter notebook D2/D2_German_Credit_FINAL.ipynb
```

### Dependencies
```
torch>=2.0
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
aif360>=0.5
```

---

## 📚 References

1. Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS 2016*.
2. Agarwal, A. et al. (2019). A reductions approach to fair classification. *ICML 2019*.
3. Hooker, S. et al. (2020). Characterising bias in compressed models. *arXiv:2010.03058*.
4. Donini, M. et al. (2018). Empirical risk minimization under fairness constraints. *NeurIPS 2018*.
5. Woodworth, B. et al. (2017). Learning non-discriminatory predictors. *NeurIPS 2017*.
6. Zafar, M.B. et al. (2017). Fairness constraints: Mechanisms for fair classification. *AISTATS 2017*.
7. Han, S. et al. (2016). Deep compression. *ICLR 2016*.
8. Kamiran, F. & Calders, T. (2012). Data preprocessing for classification without discrimination. *KAIS*.

---

## 📅 Deliverables

| Deliverable | Description | Due | Weight | Status |
|---|---|---|---|---|
| D1 | EDA + baseline ML + MLP + video | 4 June | 15% | ✅ Done |
| D2 | All methods + compression + fairness + report | 10 June | 20% | ✅ Done |
| D3 | Comparison with studies + improvement | 15 June | 15% | 🔄 In progress |
| D4 | Final report (8–10 pages) | 20 June | 25% | ⏳ Upcoming |
| D5 | Final presentation | 22 June | 25% | ⏳ Upcoming |

---

## 📄 License

This project is for academic purposes at IIT Madras Zanzibar, ML 2026 course.

---

*IIT Madras Zanzibar · Foundations of Machine Learning · 2026*
