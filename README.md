# SGCC Electricity Theft Detection Using Machine Learning

## Project Overview

This project focuses on detecting electricity theft using machine learning techniques and consumer electricity consumption data from the SGCC dataset.

The objective is to build and compare multiple classification models that identify potentially fraudulent electricity consumption patterns. Since electricity theft cases are significantly fewer than normal cases, the project addresses **class imbalance** and evaluates models using metrics beyond accuracy.

## Objectives

- Analyze and prepare the electricity consumption dataset.
- Engineer and select numerical features for classification.
- Handle missing values and class imbalance.
- Train and tune multiple machine learning models.
- Compare model performance using classification metrics.
- Visualize actual classes and model predictions using PCA.
- Analyze false negatives and identify theft cases missed by multiple models.

## Dataset

The dataset contains consumer-level electricity consumption features.

- **Target variable:** `FLAG`
  - `0` — Normal consumer
  - `1` — Potential electricity theft
- **Identifier:** `CONS_NO`
- **Features:** Engineered numerical features derived from electricity consumption data.
- **Dataset size:** 41,978 records and 29 columns in the final processed dataset.

The dataset is imbalanced, with normal consumers forming the majority class.

> **Note:** The original dataset and its redistribution terms should be checked before making the repository public.

## Machine Learning Models

The project compares five tuned classification models:

1. **Random Forest** — An ensemble of decision trees.
2. **Extra Trees** — An ensemble using randomized decision-tree splits.
3. **XGBoost** — Gradient boosting optimized for structured data.
4. **LightGBM** — A gradient-boosting framework designed for efficient training.
5. **CatBoost** — A gradient-boosting algorithm with built-in regularization and other techniques.

## Data Preprocessing

The workflow includes:

1. Separating the target variable and input features.
2. Excluding the consumer identifier from model features.
3. Splitting the data into training and test sets using stratification.
4. Imputing missing numerical values using the training-set median.
5. Standardizing features where required by the pipeline.
6. Applying SMOTE oversampling and random undersampling to address class imbalance.
7. Tuning model hyperparameters using cross-validation.

Resampling is performed within the training pipeline so that synthetic samples do not leak into the untouched test set.

## Evaluation Metrics

The models are evaluated using:

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of predicted theft cases that are actual theft cases |
| Recall | Proportion of actual theft cases detected |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Ability to distinguish between the two classes across thresholds |
| PR-AUC | Precision-recall performance, particularly useful for imbalanced datasets |
| Confusion Matrix | Counts of true positives, false positives, true negatives and false negatives |

**Recall and PR-AUC are particularly important** because failing to detect theft cases can be costly, while false alarms can lead to unnecessary investigations.

## Results

The following are the previously recorded test-set results at the models' default prediction thresholds.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9128 | 0.4698 | 0.2641 | 0.3382 | 0.7989 | 0.3582 |
| Extra Trees | 0.9148 | 0.4902 | 0.2472 | 0.3286 | 0.8028 | 0.3580 |
| XGBoost | 0.9183 | 0.5289 | 0.2839 | 0.3695 | 0.8137 | 0.3967 |
| LightGBM | 0.9190 | 0.5372 | 0.2853 | 0.3727 | 0.8176 | 0.4123 |
| CatBoost | 0.9175 | 0.5217 | 0.2542 | 0.3419 | 0.8050 | 0.3880 |

### Key Findings

- **LightGBM** achieved the highest recorded PR-AUC (0.4123), ROC-AUC (0.8176), accuracy (0.9190), precision (0.5372), recall (0.2853), and F1-score (0.3727) among these five results.
- At the default threshold, LightGBM detected 202 of the 708 theft cases in the test set.
- The results demonstrate why accuracy alone is insufficient for evaluating electricity theft detection.
- The relatively low recall indicates that substantial improvement is needed to detect more theft cases.

These are experimental results from the current notebook, not a claim that the model is ready for real-world deployment.

## Visualizations and Error Analysis

The notebook also explores model behavior through:

- PCA-based two-dimensional projections of consumer records.
- Actual-class visualizations.
- Prediction maps showing true positives, false negatives, false positives and true negatives.
- Comparisons of model predictions on the same test consumers.
- Analysis of theft cases missed by multiple models.

PCA is used for visualization; the two-dimensional projection is not a replacement for the original feature space used by the classifiers.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- XGBoost
- LightGBM
- CatBoost
- Jupyter Notebook

## Repository Structure

```text
SGCC-Electricity-Theft-Detection/
├── SGCC_Electricity_Theft_Detection.ipynb
├── README.md
├── requirements.txt
└── data/
    └── your_dataset.csv
```

Replace the example notebook and dataset filenames with the actual filenames used in your repository.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SGCC-Electricity-Theft-Detection.git
cd SGCC-Electricity-Theft-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

```bash
jupyter notebook
```

Open `SGCC_Electricity_Theft_Detection.ipynb` and execute the cells in order.

**Important:** Update the dataset path in the notebook if the CSV filename or folder location differs. Training and hyperparameter tuning may take considerable time.

## Future Improvements

- Tune classification thresholds using validation data.
- Improve recall while controlling false-positive rates.
- Investigate the characteristics of theft cases missed by every model.
- Explore additional feature engineering and model calibration.
- Evaluate performance across different consumer groups and data splits.
- Save fitted models and preprocessing objects for reproducible inference.
- Assess the model on additional unseen data before practical deployment.

## Disclaimer

This is an educational and experimental machine learning project. A model prediction alone is not proof of electricity theft. Any real-world use would require further validation, appropriate human review, and assessment of the consequences of false accusations.

## Author

Developed as a machine learning project exploring imbalanced classification and electricity theft detection.