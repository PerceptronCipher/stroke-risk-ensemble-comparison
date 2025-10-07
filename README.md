# 🏥 Stroke Risk Prediction System  

Hey there! Welcome to my stroke prediction project.  
This system predicts whether someone might be at risk of having a stroke based on their health data. Pretty exciting stuff!  

---

## What's This All About?

I grabbed this dataset from Kaggle — packed with patient details like age, smoking habits, glucose levels, and more.  
Then I applied some machine learning magic to see if we can predict stroke risk.  

**Result?** The model achieved around **77% accuracy** in spotting risk patterns.  
Not too shabby for a solo research project! ⚡  

---

## Project Structure
stroke-risk-ensemble-comparison/
│
├── data/
│ ├── raw/ # Original data lives here
│ └── processed/ # Clean, ready-to-use data
│
├── src/
│ ├── preprocessing.py # Cleans and encodes data
│ ├── train.py # Trains models and selects the best one
├
├──deployment
│ └── deploy.py # Streamlit app for live predictions
│ └── requirements.txt
│
├── models/
│ └── best_model.pkl # The winning model (Random Forest!)
│
├── run_all.py # One-click pipeline runner
└── requirements.txt # All dependencies

---

## 🚀 Getting Started

### Step 1 — Clone the Repo
```bash
git clone https://github.com/YOUR_USERNAME/stroke-risk-ensemble-comparison.git
cd stroke-risk-ensemble-comparison
```


Step 2 — Install Dependencies
pip install -r requirements.txt

Step 3 — Get the Data

Download the Healthcare Stroke Dataset from Kaggle and place it inside data/raw/.

Step 4 — Run the Pipeline
python run_all.py

Sit back and relax 😎 — the script will clean the data, train all models, and save the best performer.

Step 5 — Launch the Web App
streamlit run deployment/deploy.py
streamlit run deployment/deploy.py
You’ll have a live interactive dashboard to make real-time predictions.

How It Works
🧹 Data Preparation (preprocessing.py)

Handles missing BMI values and encodes categorical variables.

Uses SMOTE to balance the dataset (only ~5% stroke cases originally).
This ensures the model learns both classes effectively.

 Model Training (train.py)

Three powerful algorithms were compared:

Random Forest  (Champion!)

XGBoost 

LightGBM 

 Random Forest came out on top with a 0.77 ROC-AUC score —
pretty strong for medical classification!

 Deployment (deploy.py)

An interactive Streamlit app lets you:

Input patient details (age, glucose, BMI, etc.)

Get instant stroke risk predictions with confidence levels.

 Key Takeaways

Medical data is imbalanced and tricky — careful preprocessing is key.

Random Forest remains a beast even in 2025 for tabular problems.

SMOTE boosted recall for minority (stroke) cases significantly.

Balancing accuracy and safety matters — better a false positive than a missed stroke risk.

 Future Enhancements

Try neural networks for non-linear insights

Engineer new health-related features

Hyperparameter tuning with Optuna

Build an ensemble of top 3 models

My Tech Stack
| Category           | Tools                                 

| Data Handling      | `pandas`, `numpy`                     
| ML Models          | `scikit-learn`, `XGBoost`, `LightGBM` 
| Imbalance Handling | `imbalanced-learn` (SMOTE)            
| Web App            | `Streamlit`                           
| Environment        | Python 3.10+                          

🧰 Troubleshooting

All packages installed?

Data file in data/raw/?

Using Python 3.10+?

If you still hit a snag, open an issue — I’ll be glad to help.


⚠️ Disclaimer

This project is built for learning and demonstration purposes only.
It should never replace professional medical advice.
Always consult healthcare professionals for real-world decisions.

🩺 Final Thoughts

Building this was a fantastic learning experience!
Healthcare machine learning is as challenging as it is rewarding.
The model is designed to assist — not replace — medical expertise.

Stay healthy and keep coding! 

License: MIT
Dataset Credit: Kaggle — Healthcare Stroke Dataset

Built with: Passion, Python, and plenty of learning moments ❤️