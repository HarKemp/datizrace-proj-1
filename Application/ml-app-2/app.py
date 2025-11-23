import streamlit as st
import pandas as pd 
import mlflow.sklearn 
import numpy as np
from io import StringIO
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.base import BaseEstimator, TransformerMixin

class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.shape[0] > 0:
            self.lower_ = np.quantile(X, self.lower_q, axis=0)
            self.upper_ = np.quantile(X, self.upper_q, axis=0)
        else:
            self.lower_ = 0
            self.upper_ = 0
        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.clip(X, self.lower_, self.upper_)

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://172.16.238.23:5000"
MODEL_NAME = "Hotel_Cancellation_Model"
MODEL_ALIAS = "champion"

st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    layout="centered"
)

# --- MLflow Model Loading ---
@st.cache_resource
def load_mlflow_model():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        full_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        st.sidebar.info(f"Loading Model: **{MODEL_NAME}** (@{MODEL_ALIAS})")
        
        model = mlflow.sklearn.load_model(full_uri)
        st.sidebar.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading MLflow model: {e}")
        st.warning("Please ensure the MLflow server is reachable and the 'champion' alias is set.")
        return None

# --- Main App Logic ---
def main():
    st.title("Hotel Booking Cancellation Prediction")
    
    st.subheader("1. Upload Booking Data")
    st.markdown("Upload a CSV file containing hotel reservation data")
    
    # File Upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            st.subheader("2. Data Preview")
            data_df = pd.read_csv(uploaded_file)
            
            if 'Booking_ID' in data_df.columns:
                ids = data_df['Booking_ID']
                X_input = data_df.drop(['Booking_ID', 'booking_status'], axis=1, errors='ignore')
            else:
                ids = data_df.index
                X_input = data_df.drop(['booking_status'], axis=1, errors='ignore')

            st.dataframe(data_df.head(), use_container_width=True)
            st.success(f"Successfully loaded {len(data_df)} records.")

            # 3. Load Model
            model = load_mlflow_model()
            
            if model:
                # 4. Predict
                if st.button("Predict Cancellation Status"):
                    st.subheader("3. Prediction Results")
                    
                    with st.spinner("Predicting..."):
                        try:
                            predictions = model.predict(X_input)
                            
                            prediction_labels = ["Canceled" if p == 0 else "Not_Canceled" for p in predictions]

                            results_df = pd.DataFrame({
                                "Booking ID": ids,
                                "Predicted Status": prediction_labels,
                                "Raw Prediction": predictions
                            })
                            
                            # Izcelt Canceled un Not_Canceled
                            def highlight_cancel(val):
                                color = 'red' if val == 'Canceled' else 'green'
                                return f'color: {color}'

                            st.dataframe(
                                results_df.style.applymap(highlight_cancel, subset=['Predicted Status']),
                                use_container_width=True
                            )
                            
                            cancel_count = results_df[results_df['Predicted Status'] == 'Canceled'].shape[0]
                            st.metric(label="Total Predicted Cancellations", value=cancel_count)
                            
                            # Lejupielādēt csv ar rezultātiem
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Results CSV",
                                csv,
                                "predictions.csv",
                                "text/csv",
                                key='download-csv'
                            )
                            
                        except Exception as e:
                            st.error(f"Prediction failed. Error: {e}")
                            st.error("Ensure the CSV columns match the training data features.")
                            st.write("Expected columns:", list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else "Unknown")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()