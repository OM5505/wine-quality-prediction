# Wine Quality Prediction Using Machine Learning

## Overview
This project explores red wine quality data and builds a **Decision Tree Classifier** to classify wine into two groups based on its quality score:

- **0 — Poor wine:** quality score below 7
- **1 — Good wine:** quality score 7 or above

The analysis, visualizations, model training, evaluation, and sample prediction are implemented in `WineQuality.ipynb`.

## Dataset
The notebook expects a CSV file named `wineQualityReds.csv` and loads it with:

```python
df = pd.read_csv("wineQualityReds.csv")
```

Place the CSV in the same folder as the notebook. If your uploaded dataset has a different filename, update the `pd.read_csv(...)` line to match it exactly.

The notebook uses the `quality` column as the original wine-quality score and the remaining columns as model features.

## Project Workflow
1. Import Python libraries and load the dataset.
2. Inspect the dataset using summary information and descriptive statistics.
3. Check missing values and review the distribution of quality scores.
4. Visualize feature distributions and outliers with box plots.
5. Apply Z-score filtering with a threshold of 3 to remove rows identified as outliers in the feature columns.
6. Plot the resulting quality distribution and feature correlation heatmap.
7. Convert the quality score into a binary target: below 7 → 0, and 7 or above → 1.
8. Split the data into training and test sets (80% training, 20% testing; `random_state=42`).
9. Train and evaluate a Decision Tree Classifier.
10. Generate a sample prediction and estimated probability for the “good wine” class.

## Model
The notebook uses `DecisionTreeClassifier` with these settings:

- `max_depth=5`
- `min_samples_split=10`
- `min_samples_leaf=5`
- `class_weight='balanced'`
- `random_state=42`

## Evaluation
The notebook calculates and displays:
- Accuracy
- Classification report (precision, recall, and F1-score)
- Confusion matrix
- ROC curve and ROC-AUC score

Run the notebook to see the actual metric values. Results are not hard-coded in this README.

## Technologies Used
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn

## How to Run
1. Download or clone this repository.
2. Install the dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
   ```

3. Ensure the dataset CSV is in the same directory as `WineQuality.ipynb`.
4. Confirm that the filename in `pd.read_csv(...)` matches the CSV's exact name.
5. Open `WineQuality.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
6. Run the cells in order.

## Notes
- This is a learning and demonstration project; predictions are estimates from the selected dataset and model.
- The notebook's current ROC code uses the predicted class labels. For a more standard ROC-AUC calculation, use predicted probabilities (for example, `dtc.predict_proba(x_test)[:, 1]`) when calculating the ROC curve and ROC-AUC.
- Check the dataset's source and license before redistributing it publicly.
