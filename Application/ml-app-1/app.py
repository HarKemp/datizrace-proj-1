from __future__ import annotations
from io import StringIO
import os
from typing import Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
import streamlit as st

TARGET = "DRK_YN"
MODEL_NAME = "drk_classifier"
MODEL_ALIAS = "champion"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://172.16.238.23:5000")

EXPECTED_COLUMNS = [
    "sex",
    "age",
    "height",
    "weight",
    "waistline",
    "sight_left",
    "sight_right",
    "hear_left",
    "hear_right",
    "SBP",
    "DBP",
    "BLDS",
    "tot_chole",
    "HDL_chole",
    "LDL_chole",
    "triglyceride",
    "hemoglobin",
    "urine_protein",
    "serum_creatinine",
    "SGOT_AST",
    "SGOT_ALT",
    "gamma_GTP",
    "SMK_stat_type_cd",
    TARGET,
]

st.set_page_config(page_title="Drinking classification", layout="centered")
st.sidebar.title("Modeļa informācija")
st.sidebar.write(f"Modelis: `{MODEL_NAME}`")
st.sidebar.write(f"Alias: `{MODEL_ALIAS}`")


def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Validē kolonnu sarakstu un pārveido uz string tāpat kā main skriptā
    dataset = df.copy()
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Trūkst kolonnas: {missing_cols}")

    dataset[TARGET] = dataset[TARGET].astype(str)
    dataset["SMK_stat_type_cd"] = dataset["SMK_stat_type_cd"].astype(int).astype(str)
    dataset["urine_protein"] = dataset["urine_protein"].astype(int).astype(str)
    dataset["hear_left"] = dataset["hear_left"].astype(int).astype(str)
    dataset["hear_right"] = dataset["hear_right"].astype(int).astype(str)
    dataset["sex"] = dataset["sex"].astype(str)

    return dataset, dataset.index


@st.cache_resource
def load_model():
    # Ielādē MLflow modeli no champion aliasa
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    full_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    st.info(f"Ielādē modeli: {full_uri}")
    model = mlflow.sklearn.load_model(full_uri)
    st.success("Modelis ielādēts")
    return model


def main():
    st.title("Alkohola lietošanas statusa prognoze")
    st.subheader("Upload Drinking Data")
    st.markdown(
        "Augšupielādējiet CSV ar visām kolonnām"
        "Modelis tiek ielādēts no MLflow ar `champion` aliasu"
    )

    uploaded_file = st.file_uploader("1.: augšupielādējiet CSV", type="csv")
    if not uploaded_file:
        st.info("Izvēlieties datni, lai turpinātu")
        return

    try:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        data_df = pd.read_csv(stringio)
    except Exception as exc:
        st.error(f"Neizdevās nolasīt CSV: {exc}")
        return

    st.subheader("2.: datu priekšskatījums")
    st.dataframe(data_df.head(), use_container_width=True)
    st.caption(f"Kopā {len(data_df)} ieraksti.")

    try:
        dataset, indices = prepare_dataset(data_df)
    except Exception as exc:
        st.error(f"Datu sagatavošana neveiksmīga: {exc}")
        return

    model = load_model()

    if st.button("3.: prognozēšana"):
        # Prognožu solis ar rezultātu parādīšanu
        with st.spinner("Prognozē..."):
            try:
                features = dataset.drop(columns=[TARGET])
                predictions = model.predict(features)
                results_df = pd.DataFrame(
                    {"Ieraksta indekss": indices, "Prognozētais DRK_YN": predictions}
                )
                st.subheader("4.: rezultāti")
                st.dataframe(
                    results_df.style.applymap(
                        lambda val: "color: green" if val == "Y" else "color: red",
                        subset=["Prognozētais DRK_YN"],
                    ),
                    use_container_width=True,
                )

                counts = results_df["Prognozētais DRK_YN"].value_counts()
                col1, col2 = st.columns(2)
                col1.metric("Prognozētie alkohola lietotāji (Y)", counts.get("Y", 0))
                col2.metric("Prognozētie nelietotāji (N)", counts.get("N", 0))
                st.bar_chart(counts)

                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Lejupielādēt rezultātus CSV",
                    csv_bytes,
                    file_name="drk_predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Prognoze neizdevās: {exc}")


if __name__ == "__main__":
    main()
