# Liver Disease Prediction using Decision Tree Classifier

This project aims to predict liver disease in patients using a Decision Tree Classifier. The dataset used is the "Indian Liver Patient Dataset (ILPD)", and the key performance metrics such as accuracy, confusion matrix, precision, recall, and F1-score are computed to evaluate the model's performance. Visualizations such as the confusion matrix and decision tree are also generated.

## Dataset

The dataset used in this project is the **Indian Liver Patient Dataset (ILPD)**, which contains the following features:

1. `Age`: Age of the patient.
2. `Gender`: Gender of the patient (Male/Female).
3. `Total_Bilirubin`: Total Bilirubin levels.
4. `Direct_Bilirubin`: Direct Bilirubin levels.
5. `Alkphos_Alkaline_Phosphotase`: Alkaline Phosphotase levels.
6. `Sgpt_Alamine_Aminotransferase`: Alamine Aminotransferase levels.
7. `Sgot_Aspartate_Aminotransferase`: Aspartate Aminotransferase levels.
8. `Total_Proteins`: Total protein levels.
9. `Albumin`: Albumin levels.
10. `A_G_Ratio`: Albumin and Globulin ratio.
11. `Selector`: Class label (1 indicates a liver disease, 2 indicates no liver disease).

### Missing Data Handling
- Missing values in the `A_G_Ratio` column are replaced by the mean value of the column.

### Dataset Preprocessing
- The `Gender` column is encoded using a `LabelEncoder` (Male = 0, Female = 1).

## Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

You can install the required libraries using:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Running the Code

1. Clone the repository or download the code.
2. Ensure the dataset file is named `Indian Liver Patient Dataset (ILPD).csv` and is located in the same directory as the Python script.
3. Run the Python script:

```bash
python liver_disease_decision_tree.py
```

### Output

The script will output the following:
- **Accuracy**: The overall accuracy of the model.
- **Confusion Matrix**: A visual representation of the confusion matrix.
- **Classification Report**: Includes precision, recall, and F1-score for each class.
- **Decision Tree Visualization**: A visual representation of the decision-making process.

## Example Output

```
Accuracy: 0.63
Classification Report:
               precision    recall  f1-score   support

           1       0.76      0.74      0.75       129
           2       0.31      0.33      0.32        46

    accuracy                           0.63       175
   macro avg       0.53      0.54      0.53       175
weighted avg       0.64      0.63      0.64       175
```

## Visualizations

The script generates:
1. **Confusion Matrix**: Displays how well the model classifies liver disease and non-liver disease cases.
2. **Decision Tree Plot**: Helps visualize the decision path the model uses to make predictions.

## Future Improvements

- Implement other machine learning models such as Random Forest, SVM, and Logistic Regression for comparison.
- Perform hyperparameter tuning on the Decision Tree Classifier to improve accuracy.

---
