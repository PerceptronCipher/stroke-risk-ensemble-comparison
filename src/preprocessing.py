# import libraries 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os


# Load dataset
path = r"C:\Users\USER\My notebook\DataSciencePro\stroke-risk-ensemble-comparison\data\raw\healthcare-dataset-stroke-data.csv"
df = pd.read_csv(path)

# Inspect dataset
print("Dataset Info:\n")
print(df.info())
print("\n Class Distribution:\n")
print(df['stroke'].value_counts())
print("\n Sample Data:\n", df.head())

# Handle missing values 
print("Missing Values: \n", df.isnull().sum())
df.fillna({'bmi':df['bmi'].median()}, inplace=True)
print(df.info())

# Encode Categorical Data
label_encoders = {}
for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Visualize relationships
sns.pairplot(df, vars=['age', 'hypertension', 'heart_disease', 'ever_married',
                    'avg_glucose_level', 'bmi', 'smoking_status', 'stroke'])
plt.show()

# Set display options
pd.set_option('display.max_rows', None)      
pd.set_option('display.max_columns', None)   
pd.set_option('display.width', None)         
pd.set_option('display.max_colwidth', None)  

# Inspect dataset
print("Dataset Info:\n")
print(df.head())

# Scale Numerical Features 
scaler = StandardScaler()
numerical_features = ['bmi', 'avg_glucose_level', 'age']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Features and Target
X = df.drop(columns=['stroke'])
y = df['stroke']

# Split 
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Display new class distribution after smote
print("\n Class Distribution After SMOTE: \n")
print(pd.Series(y_train).value_counts())

# Save processed data
os.makedirs(r"C:\Users\USER\My notebook\DataSciencePro\stroke-risk-ensemble-comparison\data\processed", exist_ok=True)
joblib.dump((X_train, X_test, y_train, y_test), r"C:\Users\USER\My notebook\DataSciencePro\stroke-risk-ensemble-comparison\data\processed\processed_data.pkl")
