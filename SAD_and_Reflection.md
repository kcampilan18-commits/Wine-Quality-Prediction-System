# SAD Mini-Analysis and Reflection
## Wine Quality Prediction System – Section B

### 1. SAD mini-analysis

**Problem statement:** A wine producer needs a consistent preliminary way to
screen wine samples using recorded physicochemical measurements.

**Users:** Quality-control staff, production supervisor, or data analyst.

**Inputs:** Fixed acidity, volatile acidity, citric acid, residual sugar,
chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates,
and alcohol.

**Process:** Validate input columns → preprocess data → apply trained model →
classify quality category.

**Outputs:** `GOOD QUALITY` or `NEEDS IMPROVEMENT` label and model evaluation
summary.

**Key limitation:** The model is trained on a particular red-wine dataset and
may not generalize to other wine types, producers, regions, or testing
conditions.

### 2. Brief reflection answers

**1. What is the prediction target, and why is the original quality column
excluded from the input features?**

The prediction target is the binary wine-quality category: GOOD QUALITY when
the original score is 6 or higher, and NEEDS IMPROVEMENT when it is below 6.
The original `quality` column is excluded from the input features because it
is used to derive the target. Including it as an input would cause target
leakage.

**2. What does recall for the GOOD QUALITY class tell you? What could low
recall mean for the business?**

Recall for the GOOD QUALITY class tells us what proportion of all actual
good-quality samples were correctly identified by the model. Low recall means
some good-quality samples are incorrectly classified as needing improvement,
which could lead to unnecessary additional review or missed opportunities.

**3. Why might a model trained on this red-wine dataset perform differently
on another producer's wine samples?**

Another producer may use different grapes, ingredients, production methods,
or conditions. These differences can change the physicochemical measurements,
so the model may not perform the same way on those samples.

**4. Why should the prediction be treated as a preliminary screening result
rather than a final quality decision?**

The system is a classroom prototype trained on a particular public dataset.
Its prediction does not replace laboratory testing, sensory evaluation,
food-safety procedures, or professional quality-control decisions.

**5. What additional data or testing would help improve the reliability of
the system?**

More representative samples from different producers, regions, and batches
could help. The model should also be tested on independent samples and
evaluated using appropriate validation methods before being used in real
quality-control work.

### 3. Evaluation results – fill in after running the model

Do not invent or estimate these values. Run `python train_model.py` and copy
the actual output into this table.

| Metric | GOOD QUALITY | NEEDS IMPROVEMENT |
|---|---:|---:|
| Precision | __________ | __________ |
| Recall | __________ | __________ |
| F1-score | __________ | __________ |

**Overall accuracy:** __________

**Confusion matrix:** Copy the 2×2 matrix printed by the training script.

### 4. Screenshot evidence to capture

1. **Dashboard screenshot:** Open the running Streamlit app and capture the
   title, dataset metrics, and dataset preview.
2. **Prediction screenshot:** Upload `sample_wine_input.csv` and capture the
   prediction results and predicted-class counts.
3. **Optional test screenshot:** Upload a CSV with one required column removed
   and capture the helpful missing-column error.

Use real screenshots from your own running application. Do not submit mock or
fabricated screenshots.

### 5. GitHub repository

Paste your actual public repository URL here after pushing the project:

`https://github.com/YOUR-USERNAME/wine-quality-prediction`
