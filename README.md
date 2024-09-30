This code performs stroke prediction using logistic regression and K-nearest neighbors (KNN) classifiers. Here's an explanation of each part of the code:

1. **Importing Libraries**:
   - `pandas`: For data manipulation and analysis.
   - `numpy`: For numerical computing.
   - `seaborn`, `matplotlib.pyplot`: For data visualization.
   - `train_test_split`: For splitting the dataset into training and testing sets.
   - `KNeighborsClassifier`: For implementing the KNN classifier.
   - `accuracy_score`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`: For evaluating model performance.
   - `RandomOverSampler`, `RandomUnderSampler`: For handling class imbalance.
   - `LabelEncoder`: For encoding categorical variables.
   - `tabulate`: For displaying data in a tabular format.
   - `LogisticRegression_imp`: Custom implementation of logistic regression.
   - `tkinter`, `ttk`, `messagebox`: For creating a graphical user interface (GUI).

2. **Custom Logistic Regression Implementation** (`LogisticRegression_imp` class):
   - This class implements logistic regression using gradient descent.
   - It includes methods for fitting the model to the training data, making predictions, and evaluating model performance.

3. **Data Preprocessing**:
   - The dataset is loaded using `pd.read_csv`.
   - Categorical variables are encoded using `LabelEncoder`.
   - Missing values in the 'bmi' column are replaced with the median value.

4. **Model Training and Evaluation**:
   - The dataset is split into training and testing sets using `train_test_split`.
   - A KNN classifier is trained and evaluated on the dataset.
   - Random over-sampling and under-sampling techniques are applied to handle class imbalance.
   - Logistic regression is trained and evaluated using the custom implementation.

5. **GUI Implementation** (`StrokePredictionGUI` class):
   - This class creates a graphical user interface for the stroke prediction system.
   - It includes input fields for age, BMI, and smoking status, and a button to predict stroke risk.
   - When the predict button is clicked, the input values are retrieved, and a prediction is made based on the custom logic.
   - Error handling is included to handle invalid input values.

6. **Main Function**:
   - The main function creates an instance of the `StrokePredictionGUI` class and starts the GUI application using `tkinter`.

Overall, this code demonstrates how to build a stroke prediction system using logistic regression and KNN classifiers and provides a simple GUI for user interaction.

![GitHub Image](/images/cerebal stroke)
