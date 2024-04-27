import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

class LogisticRegression_imp:
    def __init__(self, learning_rate=0.01, num_iterations=1000, threshold=0.5):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
        self.losses = []  # List to store loss values during training

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _standardize(self, X, fit=False):
        if fit:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        X_standardized = (X - self.mean) / self.std
        return X_standardized

    def _initialize_parameters(self, num_features):
        self.weights = np.zeros(num_features)
        self.bias = 0

    def fit(self, X, y):
        X = self._standardize(X, fit=True)
        num_samples, num_features = X.shape
        self._initialize_parameters(num_features)
        W1 = y.shape[0] / (2 * y.sum())
        W0 = y.shape[0] / (2 * (y.shape[0] - y.sum()))

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_model)

            # Calculate loss
            loss = (-1 / num_samples) * np.sum(
                W1 * y * np.log(predictions) + W0 * (1 - y) * np.log(1 - predictions)
            )

            # Append loss to list for plotting
            self.losses.append(loss)

            # Gradient descent
            dw = (1 / num_samples) * np.dot(X.T, -(W1 * y * (1 - predictions)) + W0 * (1 - y) * predictions)
            db = (1 / num_samples) * np.sum(-(W1 * y * (1 - predictions)) + W0 * (1 - y) * predictions)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        # Plot loss curve
        plt.plot(range(1, self.num_iterations + 1), self.losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.show()

        # Print final loss value
        print('Final Loss:', self.losses[-1])

    def predict(self, X):
        X = self._standardize(X)
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(linear_model)
        return np.where(predictions >= self.threshold, 1, 0)

    def evaluate(self, X, y):
        X = self._standardize(X)
        y_pred = self.predict(X)
        # Calculate accuracy
        accuracy = accuracy_score(y, y_pred)
        print("Accuracy:", accuracy)

        # Calculate precision
        precision = precision_score(y, y_pred)
        print("Precision:", precision)

        # Calculate recall
        recall = recall_score(y, y_pred)
        print("Recall:", recall)

        # Calculate F1 score
        f1 = f1_score(y, y_pred)
        print("F1 Score:", f1)

        # Generate confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)

        # Plot confusion matrix using seaborn
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # plot pie chart
        labels = ['Not Stroke', 'Stroke']
        sizes = [y_pred.shape[0] - y_pred.sum(), y_pred.sum()]
        explode = (0, 0.1)

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Predicted Class Distribution')
        plt.show()

# Load data
data = pd.read_csv('dataset.csv')

# Preprocessing
data['gender'] = LabelEncoder().fit_transform(data['gender'])
data['Residence_type'] = LabelEncoder().fit_transform(data['Residence_type'])
data['ever_married'] = LabelEncoder().fit_transform(data['ever_married'])
data['smoking_status'] = LabelEncoder().fit_transform(data['smoking_status'])
data = data.drop(['work_type'], axis=1)
data['bmi'].fillna(data['bmi'].median(), inplace=True)

# Splitting the data
X = data.drop(['stroke'], axis=1)
Y = data['stroke']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)

# Applying KNN model
KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=1)
KNN.fit(X_train, Y_train)
Y_pred = KNN.predict(X_test)

# Accuracy of KNN model
acc = accuracy_score(Y_test, Y_pred)
print("The accuracy of the KNN model is %.2f" % (acc * 100), "%")
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(cm)

# Applying Random Over-sampling
ros = RandomOverSampler()
X_resample1, Y_resample1 = ros.fit_resample(X_train, Y_train)
KNNov = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=1)
KNNov.fit(X_resample1, Y_resample1)
Y_pred1 = KNNov.predict(X_test)
acc1 = accuracy_score(Y_test, Y_pred1)
print("The accuracy of the Over-sampled KNN model is %.2f" % (acc1 * 100), "%")
cm1 = confusion_matrix(Y_test, Y_pred1)
print("Confusion Matrix after Random Over-sampling:")
print(cm1)

# Applying Random Under-sampling
rus = RandomUnderSampler()
X_resample2, Y_resample2 = rus.fit_resample(X_train, Y_train)
KNNun = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=1)
KNNun.fit(X_resample2, Y_resample2)
Y_pred2 = KNNun.predict(X_test)
acc2 = accuracy_score(Y_test, Y_pred2)
print("The accuracy of the Under-sampled KNN model is %.2f" % (acc2 * 100), "%")
cm2 = confusion_matrix(Y_test, Y_pred2)
print("Confusion Matrix after Random Under-sampling:")
print(cm2)

# Applying Logistic Regression
logistic_reg = LogisticRegression_imp()
logistic_reg.fit(X_train.values, Y_train.values)
logistic_reg.evaluate(X_test.values, Y_test.values)


class StrokePredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stroke Prediction")

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)

        self.age_label = ttk.Label(self.main_frame, text="Age:")
        self.age_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.age_entry = ttk.Entry(self.main_frame)
        self.age_entry.grid(row=0, column=1, padx=5, pady=5)

        self.bmi_label = ttk.Label(self.main_frame, text="BMI:")
        self.bmi_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.bmi_entry = ttk.Entry(self.main_frame)
        self.bmi_entry.grid(row=1, column=1, padx=5, pady=5)

        self.smoking_status_label = ttk.Label(self.main_frame, text="Smoking Status:")
        self.smoking_status_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.smoking_status_entry = ttk.Entry(self.main_frame)
        self.smoking_status_entry.grid(row=2, column=1, padx=5, pady=5)

        self.predict_button = ttk.Button(self.main_frame, text="Predict", command=self.predict_stroke)
        self.predict_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

    def predict_stroke(self):
        try:
                age = float(self.age_entry.get())
                bmi = float(self.bmi_entry.get())
                smoking_status = int(self.smoking_status_entry.get())  # Corrected to int

                # Mock prediction based on the provided data
                if age > 50 and bmi > 30 and smoking_status == 1:
                    prediction = "High Risk of Stroke"
                else:
                    prediction = "Low Risk of Stroke"

                messagebox.showinfo("Stroke Prediction Result", f"The predicted risk of stroke is: {prediction}")
    
        except ValueError:
                messagebox.showerror("Error", "Please enter valid numerical values for Age, BMI, and Smoking Status")



if __name__ == "__main__":
    root = tk.Tk()
    app = StrokePredictionGUI(root)
    root.mainloop()
