import streamlit as st
import pandas as pd 
import mlflow.sklearn 
from sklearn.preprocessing import LabelEncoder
import numpy as np
from io import StringIO


st.set_page_config(
    page_title="Aizdevuma statusa prognozēšanas modelis",
    layout="centered"
)

# --- Funkcija datu sagatavošanai ---
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    
    st.info("Notiek datu kopas sagatavošana")
    
    dataset = df.copy()
    
    # Nosakam skaitliskās kolonnas
    numerical_cols = dataset.select_dtypes(include=['int64','float64']).columns.tolist()
    # Nosakam kategoriju kolonnas
    categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()
    # Izņemsim kolonnas, kuras mūs neinteresē
    categorical_cols.remove('Loan_Status')
    categorical_cols.remove('Loan_ID')

    # Ar modu aizpildam tukšas vērtības kategoriju kolonnās
    for col in categorical_cols:
        dataset[col] = dataset[col].fillna(dataset[col].mode()[0])

    # Ar mediānu aizpildām tukšās vērtības skaitliskajās kolonnās
    for col in numerical_cols:
        dataset[col] = dataset[col].fillna(dataset[col].median(skipna=True))
    
    #Skaitlisko kolonnu normalizācija un kopējā ienakuma aprēķins
    dataset['LoanAmount'] = np.log(dataset['LoanAmount']).copy()
    dataset['TotalIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
    dataset['TotalIncome'] = np.log(dataset['TotalIncome']).copy()

    # Nevajadzīgo kolonnu nodzēšana
    dataset = dataset.drop(columns=['ApplicantIncome','CoapplicantIncome'])

    # Kategoriju atribūtu vērtību iekodēšana ar skaitļiem. Tiek pielietots LabelEncoder, kurš to dara automātiski.
    for col in categorical_cols:
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])

    dataset = dataset.drop(columns=['Loan_Status', 'Loan_ID'])
    return dataset

# --- MLflow Model Loading with Caching ---
@st.cache_resource
def load_mlflow_model():
    try:
        # Full model URI format: models:/<model_name>/<version_alias>
        MODEL_NAME = "rf_champion"
        MODEL_ALIAS = "champion"
        full_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        st.info(f"Ielādējam modeli: **{full_uri}**")
        
        model = mlflow.sklearn.load_model(full_uri)
        st.success("Modelis tika veiksmīgi ielādēts.")
        return model
    except Exception as e:
        st.error(f"Error loading MLflow model: {e}")
        st.warning("Ensure your MLFLOW_TRACKING_URI is correct and the model 'rf_champion' with alias 'champion' exists.")
        # Return the placeholder model on failure so the app can still demonstrate the flow
        return model


# --- Main Streamlit App Logic ---
def main():
    st.title("Aizdevuma statusa prognozēšanas modelis")
    
    st.subheader("1. Augšupielādējiet datni ar datiem")
    st.markdown("Augšupielādējiet datni ar datiem, kur sākuma rinda satur kolonnu sarakstu.")
    
    # Augšupielāde
    uploaded_file = st.file_uploader(
        "Izvēlietise CSV datni",
        type="csv"
    )

    if uploaded_file is not None:
        try:
            # 2. Load data into DataFrame
            st.subheader("2. Datu ielāde")
            
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            data_df = pd.read_csv(stringio)
            
            st.dataframe(data_df.head(), use_container_width=True)
            st.success(f"Veiksmīgi ielādēti {len(data_df)} ieraksti.")

            # 3. Preprocess the Data
            st.subheader("3. Datu sagatavošana")
            preprocessed_df = preprocess_data(data_df)
            
            st.dataframe(preprocessed_df.head(), use_container_width=True)
            st.success("Datu sagatavošana pabeigta.")
            
            # 4. Load the Model
            st.subheader("4. Aizdevuma statusu prognozēšana")
            
            # Load the model (cached)
            model = load_mlflow_model()
            
            if model is not None:
                st.info("Prognozējam vērtības...")
                
                # 5. Predict target feature values
                predictions = model.predict(preprocessed_df)
                
                # 6. Prepare and Display Results
                results_df = pd.DataFrame({
                    "Record Index": data_df.Loan_ID,
                    "Forecasted Value": predictions
                })
                
                st.subheader("5. Rezultāti")
                st.dataframe(results_df, use_container_width=True)
                st.success("Prognozēšana pabeigta un rezultāti parādīti!")
                
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.warning("Please ensure your CSV file is correctly formatted and contains the expected features for the ML model.")

if __name__ == "__main__":
    main()

