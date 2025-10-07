import subprocess
import sys

print("="*60)
print("STROKE RISK PREDICTION - FULL PIPELINE")
print("="*60)

# Step 1: Run preprocessing
print("\n[STEP 1/2] Running preprocessing...")
result = subprocess.run([sys.executable, "src/preprocessing.py"], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("ERROR in preprocessing:", result.stderr)
    sys.exit(1)

# Step 2: Run training
print("\n[STEP 2/2] Running model training...")
result = subprocess.run([sys.executable, "src/train.py"], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("ERROR in training:", result.stderr)
    sys.exit(1)

print("\n" + "="*60)
print("PIPELINE COMPLETE!")
print("="*60)
print("Processed data saved to: data/processed/")
print("Best model saved to: models/best_model.pkl")
print("Run deployment: streamlit run src/deploy.py")