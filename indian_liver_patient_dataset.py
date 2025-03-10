# -*- coding: utf-8 -*-
"""Indian Liver Patient Dataset

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YqfRftzO7dbTbdr0DiFQZC5jmtz58Gt1
"""

# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_path = '/content/Indian Liver Patient Dataset (ILPD).csv'
data = pd.read_csv(file_path)

column_names = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
    'Alkphos_Alkaline_Phosphotase', 'Sgpt_Alamine_Aminotransferase',
    'Sgot_Aspartate_Aminotransferase', 'Total_Proteins', 'Albumin',
    'A_G_Ratio', 'Selector'
]
data.columns = column_names

data['A_G_Ratio'].fillna(data['A_G_Ratio'].mean(), inplace=True)

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

X = data.drop('Selector', axis=1)
y = data['Selector']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

class_report = classification_report(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualizing the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No Liver Disease', 'Liver Disease'], rounded=True)
plt.title('Decision Tree Visualization')
plt.show()

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", class_report)