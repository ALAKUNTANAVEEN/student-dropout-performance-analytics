****Student Dropout & Performance Analytics****

This project applies predictive analytics and machine learning to identify students at risk of dropping out and to predict final academic performance. The system integrates multiple data sources and deploys trained models through an interactive Streamlit web application for practical use by academic advisors and institutions.


**🚀 Project Objectives**

Predict student dropout risk using classification models.
Predict final student grades (G3) using regression models.
Prioritize recall to minimize missed at-risk students.
Deploy models in a user-friendly dashboard for real-world decision support.


**📊 Datasets Used**

1.Portuguese Higher Education Dataset (2008–2018)
~4,424 students, 35 features
Target: Graduate / Enrolled / Dropout
Used for dropout classification

2.UCI Student Performance Dataset – Portuguese (student-por.csv)
649 students, 33 features
Target: Final Grade (G3)
Used for regression

Features include academic indicators, socio-economic factors, and behavioral aspects.


**🧠 Models Implemented**

Classification (Dropout Prediction)
Logistic Regression (baseline)
Decision Tree
Random Forest (best performer, deployed)
HistGradientBoosting (comparative)
Regression (Final Grade Prediction)
Linear Regression
ElasticNet Regression


**📈 Model Evaluation**

70/30 Train-Test Split
10-Fold Cross-Validation

Metrics:
Accuracy, Precision, Recall, F1-Score
ROC-AUC, Precision-Recall Curve
Threshold tuning using F2-score to prioritize recall


**🖥️ Streamlit Application**

The dashboard provides:
Quick Mode and Accurate Mode predictions
Adjustable risk thresholds (Balanced / High Recall / High Precision)
JSON input/output for testing and reproducibility

Separate tabs for:
Dropout Risk Prediction
Final Grade (G3) Prediction


**⚖️ Ethical Considerations**

Student data is anonymized
Predictions are intended for support and early intervention, not penalization
Emphasis on fairness, transparency, and interpretability


**🛠️ Tech Stack**

Python, Pandas, NumPy
Scikit-learn
Streamlit
Joblib


**📌 Project Status**

✅ Completed – Submitted as part of MSc Dissertation
🎓 Program: MSc in Big Data Management and Analytics
🏫 Institution: Griffith College, Dublin

