import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class StrokePredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stroke Prediction")

        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0)

        # Age
        self.age_label = ttk.Label(self.main_frame, text="Age:", font=("Helvetica", 12, "bold"))
        self.age_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.age_entry = ttk.Entry(self.main_frame, font=("Helvetica", 12))
        self.age_entry.grid(row=0, column=1, padx=5, pady=5)

        # BMI
        self.bmi_label = ttk.Label(self.main_frame, text="BMI:", font=("Helvetica", 12, "bold"))
        self.bmi_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.bmi_entry = ttk.Entry(self.main_frame, font=("Helvetica", 12))
        self.bmi_entry.grid(row=1, column=1, padx=5, pady=5)

        # Smoking Status
        self.smoking_status_label = ttk.Label(self.main_frame, text="Smoking Status:", font=("Helvetica", 12, "bold"))
        self.smoking_status_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.smoking_status_entry = ttk.Entry(self.main_frame, font=("Helvetica", 12))
        self.smoking_status_entry.grid(row=2, column=1, padx=5, pady=5)

        # Predict Button
        self.predict_button = ttk.Button(self.main_frame, text="Predict", command=self.predict_stroke)
        self.predict_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

    def predict_stroke(self):
        try:
            age = float(self.age_entry.get())
            bmi = float(self.bmi_entry.get())
            smoking_status = int(self.smoking_status_entry.get())

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
