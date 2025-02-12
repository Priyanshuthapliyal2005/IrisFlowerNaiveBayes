# Iris Flower Classification Using Naive Bayes

## Overview
This project demonstrates how to classify Iris flowers using the Naive Bayes algorithm. The notebook involves data handling with Pandas, directory management, and implementing machine learning techniques for classification.

---

## Code Explanation (Line-by-Line)

### 1. Importing Required Libraries
```python
import pandas as pd
import os
```
- **`import pandas as pd`**: Imports the Pandas library and aliases it as `pd`. Pandas is essential for data manipulation and analysis.
- **`import os`**: Imports the OS module, allowing interaction with the operating system (like file and directory operations).

### 2. Displaying the Current Working Directory
```python
%pwd
```
- **`%pwd`**: A magic command in Jupyter notebooks to display the current working directory.

### 3. Creating a New Directory
```python
!mkdir Iris
```
- **`!mkdir Iris`**: Executes a shell command to create a new directory named `Iris`. The `!` allows shell commands within the notebook.
- **Note**: The output "File exists" indicates that the directory already exists.

### 4. Loading the Dataset
```python
from sklearn.datasets import load_iris
data = load_iris()
```
- **`from sklearn.datasets import load_iris`**: Imports the Iris dataset loader from Scikit-learn.
- **`data = load_iris()`**: Loads the Iris dataset into the variable `data`.

### 5. Converting Dataset to a DataFrame
```python
df = pd.DataFrame(data.data, columns=data.feature_names)
```
- **`pd.DataFrame(data.data, columns=data.feature_names)`**: Converts the dataset into a Pandas DataFrame for easier data manipulation.
- **`df`**: The DataFrame that now holds the Iris dataset with appropriate column names.

### 6. Adding Target Column
```python
df['target'] = data.target
```
- **`df['target'] = data.target`**: Adds a new column named `target` containing the classification labels for each Iris flower.

### 7. Displaying the First Few Rows
```python
df.head()
```
- **`df.head()`**: Displays the first five rows of the DataFrame to give a preview of the dataset.

### 8. Importing Naive Bayes Classifier
```python
from sklearn.naive_bayes import GaussianNB
```
- **`from sklearn.naive_bayes import GaussianNB`**: Imports the Gaussian Naive Bayes classifier from Scikit-learn, suitable for continuous data.

### 9. Initializing the Model
```python
model = GaussianNB()
```
- **`GaussianNB()`**: Initializes the Naive Bayes classifier.
- **`model`**: The variable holding the initialized Naive Bayes model.

### 10. Splitting Features and Target
```python
X = df.drop('target', axis=1)
y = df['target']
```
- **`df.drop('target', axis=1)`**: Drops the `target` column to separate the features (stored in `X`).
- **`df['target']`**: Selects the `target` column to be used as the dependent variable (stored in `y`).

### 11. Fitting the Model
```python
model.fit(X, y)
```
- **`model.fit(X, y)`**: Trains the Naive Bayes classifier using the feature data `X` and the target labels `y`.

### 12. Making Predictions
```python
predictions = model.predict(X)
```
- **`model.predict(X)`**: Uses the trained model to make predictions on the dataset.
- **`predictions`**: Stores the predicted class labels.

### 13. Evaluating the Model
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y, predictions)
```
- **`from sklearn.metrics import accuracy_score`**: Imports the accuracy score function from Scikit-learn.
- **`accuracy_score(y, predictions)`**: Calculates the accuracy of the model by comparing the true labels `y` with the predicted labels.
- **`accuracy`**: Stores the accuracy score of the model.

### 14. Displaying the Accuracy
```python
print(f"Accuracy: {accuracy}")
```
- **`print(f"Accuracy: {accuracy}")`**: Prints the accuracy score in a formatted string.

---

## Dataset Handling
The dataset is loaded from Scikit-learn's built-in datasets, converted into a Pandas DataFrame, and pre-processed by separating features and labels.

---

## Conclusion
The notebook walks through the necessary steps to classify Iris flowers using Naive Bayes, starting from data preparation to model evaluation. Each step is clearly defined to help understand the workflow.

This detailed breakdown helps in understanding the code logic, making it easier to modify and apply to similar classification problems.

