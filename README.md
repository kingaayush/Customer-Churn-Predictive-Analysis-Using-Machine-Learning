This repository contains a complete solution for predicting customer retention risk and generating business-friendly insights using Python-based machine learning integrated into Power BI. It works dynamically with both labeled (Churn column present) and unlabeled datasets.

🚀 How to Use This Project
🔧 Step 1: Clone or Download the Repository
Clone the repository or download the ZIP and extract it to your local machine.

Ensure all the necessary files (train_churn_model.py, model_script.py, saved .pkl files, and the Power BI report) are in a single directory.

📁 Step 2: Train the Model on Your Dataset (If You Wish to Retrain)
Open the train_model.py script.

Replace the placeholder file name in the following line with your training dataset:

df = pd.read_csv("churn_data2.csv")  # 🔄 Replace with your file name
Save the script and run it.
It will:

Clean the dataset

Encode necessary features

Train a Random Forest model

Save the trained model (churn_model.pkl), label encoders, and expected feature list

🔄 Step 3: Load Model in Power BI
You will use Python scripting in Power BI to run the model and visualize results.

a) Load Source Data
In Power BI, go to Home > Get Data > Text/CSV and import the dataset you want to analyze.

In the Transform Data (Power Query Editor) step, rename the dataset to a simple name (e.g., dataset).

b) Replace the Source Python Code
Click on the gear icon next to the dataset’s source step (or right-click the applied step > Edit Source).

Replace the default code with this:

import pandas as pd  
dataset = pd.read_csv("YourFileName.csv")  # Replace with your actual file
✅ Be sure to replace "YourFileName.csv" with the exact name of the file you're analyzing.

🧠 Step 4: Apply the Model Script in Power BI
In the Power Query Editor, go to Transform > Run Python Script.

Paste the contents of model_script.py there.

Power BI will:

Apply the trained model

Output predictions (Predicted_Churn, Churn_Probability)

If actual churn values exist in your dataset, it will calculate evaluation metrics

📈 Step 5: Visualize the Results in Power BI
Once data loads back to Power BI:

Use Python Visuals to display custom plots like:

Confusion Matrix

ROC Curve

Feature Importance

Churn by Contract Type

Probability Distributions

Evaluation Metrics

Lift Chart

✅ You can choose which visualizations to display depending on whether your dataset has an actual Churn column or not.

📂 Included Files
File	Description
train_model.py	Script to preprocess data and train a Random Forest model
model_script.py	Prediction script for use inside Power BI
churn_model.pkl	Saved trained model
label_encoders.pkl	Encoded labels for categorical columns
expected_features.pkl	List of features expected during prediction
PowerBI Report.pbix	Final interactive Power BI dashboard with visualizations

📌 Notes
This setup supports both labeled and unlabeled datasets.

You can modify the prediction threshold (0.3 by default) inside model_script.py to suit your sensitivity/recall needs.

Python and required libraries (pandas, numpy, scikit-learn, joblib, matplotlib, seaborn) must be installed on the system for Power BI Python visuals to function.

💡 Example Use Case
Upload a customer transaction dataset to Power BI, and this solution will return:

Customers most likely to leave

Factors influencing customer behavior

Data-driven visualizations that aid retention planning
