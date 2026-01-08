🧠 Stroke Risk Prediction System

A machine learning–powered system that predicts an individual’s likelihood of stroke occurrence based on key clinical and lifestyle indicators. This project applies ensemble algorithms to healthcare data to identify high-risk patients and support early medical intervention.

📘 Overview

Early prediction of stroke risk can significantly reduce mortality and improve preventive care.
This project leverages patient health metrics such as age, BMI, glucose levels, and smoking habits to develop a data-driven stroke risk classifier.

Using an ensemble comparison framework, multiple algorithms were trained, evaluated, and benchmarked to identify the best-performing model.

Result:
The Random Forest classifier achieved a ROC-AUC score of 0.77, outperforming other ensemble models (XGBoost, LightGBM, CatBoost, Gradient Boosting).

🏗️ Project Structure
stroke-risk-ensemble-comparison/
```
├── data/
│   ├── raw/           # Original dataset
│   └── processed/     # Cleaned and transformed data
│
├── src/
│   ├── preprocessing.py    # Handles cleaning, encoding, and SMOTE balancing
│   └── train.py            # Trains and compares all ensemble models
│
├── deployment/
│   ├── deploy.py           # Streamlit web interface for live predictions
│   └── requirements.txt
│
├── models/
│   └── best_model.pkl      # Saved Random Forest model
│
├── run_all.py              # Automated end-to-end pipeline
└── requirements.txt        # Dependency list
```

⚙️ Getting Started
1️⃣ Clone the Repository
git clone https://github.com/PerceptronCipher/stroke-risk-ensemble-comparison.git
cd stroke-risk-ensemble-comparison


2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Add the Dataset
data/raw/

4️⃣ Run the Complete Pipeline
python run_all.py
This will clean the dataset, balance it using SMOTE, train all models, and store the best performer.

5️⃣ Launch the Web App
streamlit run deployment/deploy.py


🔍 Methodology
1. Data Preparation

- Handled missing BMI values and outliers

- Encoded categorical variables

- Applied SMOTE to balance minority (stroke) cases

2. Model Training & Evaluation

- Compared five ensemble models:

- Random Forest → Best ROC-AUC = 0.77

- XGBoost

- LightGBM

- Gradient Boosting

- CatBoost

Each model was evaluated on ROC-AUC, precision, recall, and f1-score to determine robustness and clinical reliability.

3. Deployment

- Built a Streamlit web application that enables:

- Input of patient parameters (age, glucose, BMI, etc.)

- Instant stroke risk prediction with probability output

🧩 Key Insights

- Addressing class imbalance is crucial for medical data (SMOTE improved recall).

- Random Forest remains a strong baseline for tabular healthcare data.

- In risk prediction, recall is often more important than overall accuracy — it’s better to flag a potential risk than to miss one.

- Feature scaling and careful preprocessing significantly impact medical model validity.

🚀 Future Improvements

- Hyperparameter optimization using Optuna

- Feature engineering with domain-specific health variables

- Ensemble stacking for performance boosting

- Integration with cloud-based MLOps for scalable deployment

🧰 Tech Stack
```
| Category           | Tools / Frameworks                                
 
| Data Handling      | `pandas`, `numpy`                                 
| ML Algorithms      | `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost` 
| Imbalance Handling | `imbalanced-learn (SMOTE)`                        
| Visualization / UI | `Streamlit`                                       
| Environment        | `Python 3.10+`                                    
```

⚠️ Disclaimer

This project is intended for educational and research purposes only.
It should not be used for clinical decision-making or as a substitute for professional medical evaluation.

📜 License

MIT License

📊 Dataset Source

Kaggle — Healthcare Stroke Dataset

💡 Author

Boluwatife Adeyemi
AI Engineer | Machine Learning Researcher | Data Scientist
📫 adeyemiboluwatife.olayinka@gmail.com

🔗 GitHub Profile