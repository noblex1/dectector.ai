
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

def train_and_save_model():
    print("Training model since model.pkl was not found...")
    df = pd.read_csv("nsl_kdd_sample.csv")
    X = df.select_dtypes(include=['int64', 'float64'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(X_scaled)
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model and scaler saved.")

if __name__ == "__main__":
    train_and_save_model()
