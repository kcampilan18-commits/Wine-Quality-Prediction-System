# Wine Quality Prediction System

**Section B – Data Mining Laboratory**

A classroom prototype that classifies red-wine samples as **GOOD QUALITY** or
**NEEDS IMPROVEMENT** using physicochemical measurements and a Random Forest
classifier.

## Prediction rule

- Original quality score **6 or higher** → `GOOD QUALITY` (class 1)
- Original quality score **below 6** → `NEEDS IMPROVEMENT` (class 0)

The original `quality` column is used only to create the target. It is excluded
from the model input features to prevent target leakage.

## Dataset

UCI Machine Learning Repository – Wine Quality:
https://archive.ics.uci.edu/dataset/186/wine+quality

The training script downloads the red-wine CSV automatically. Internet access
is required when running `train_model.py` for the first time.

## Technologies

- Python 3.10 or later
- pandas
- scikit-learn
- joblib
- Streamlit

## Project files

```text
wine-quality-prediction/
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── SAD_and_Reflection.md
```

Running the training script also creates:
- `wine_quality.csv` – downloaded dataset
- `wine_model.pkl` – trained model pipeline
- `evaluation_results.json` – test-set evaluation metrics

## Installation and running (Windows / VS Code)

1. Install Python 3.10 or later.
2. Open this folder in VS Code.
3. Open Terminal → New Terminal.
4. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

5. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

6. Train and evaluate the model:

   ```powershell
   python train_model.py
   ```

7. Start the Streamlit app:

   ```powershell
   streamlit run app.py
   ```

8. Open the local URL shown in the terminal (normally
   `http://localhost:8501`).
9. In the app, download `sample_wine_input.csv`, upload it, inspect the
   predictions, and download the prediction-results CSV.

## Evaluation

The training script prints accuracy, precision, recall, and F1-score for both
classes, plus a confusion matrix and classification report. It also saves the
numeric results in `evaluation_results.json`.

**Important:** Do not write invented metric values in your report. Run
`python train_model.py` and copy the actual values printed by your run.

## GitHub publishing

Create a public GitHub repository named `wine-quality-prediction`. From this
project folder, run:

```bash
git init
git add train_model.py app.py requirements.txt README.md SAD_and_Reflection.md
git commit -m "Build wine quality prediction prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/wine-quality-prediction.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Do not put passwords or
access tokens in source code. Add generated model/data files only if your
instructor permits them.

## Limitations and responsible use

This model is trained on the UCI red-wine dataset and may not generalize to
other wine types, producers, regions, or testing conditions. A prediction is
not a substitute for laboratory testing, sensory evaluation, food-safety
procedures, or professional quality-control decisions.
