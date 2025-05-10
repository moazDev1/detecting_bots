# VK Bot Detection Project

This project builds a machine learning pipeline to detect bot accounts on VK.com using structured profile-level data (no text or network analysis).

## 📌 What We Did

- **Data Preprocessing**
  - Filled missing numeric values with median and categorical values with mode.
  - Normalized numeric features using Min-Max Scaling.
  - One-hot encoded categorical features.
  - Saved processed train/test sets.

- **Model Training**
  - Trained four models: Naive Bayes, Decision Tree, Logistic Regression, and Random Forest.
  - Saved models using `joblib` for reuse.

- **Evaluation**
  - Measured accuracy, precision, recall, F1 score, and generated confusion matrices.
  - Visualized ROC curve and top features.
  - Random Forest showed the best test performance (~96.5% accuracy).

- **Visualization**
  - Created graphs for model comparison, ROC curve, feature importance, and confusion matrices.

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install pandas scikit-learn matplotlib joblib
python main.py
