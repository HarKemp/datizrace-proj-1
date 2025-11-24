from __future__ import annotations

from io import StringIO
import os

import mlflow.sklearn
import pandas as pd
import streamlit as st

TARGET = "DRK_YN"
MODEL_NAME = "drk_classifier"
MODEL_ALIAS = "champion"

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
    "DRK_YN",
]

st.set_page_config(page_title="Drinking Classification", layout="centered")


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    dataset = df.copy()
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Trūkst kolonnas: {missing_cols}")

    dataset["SMK_stat_type_cd"] = dataset["SMK_stat_type_cd"].astype(int).astype(object)
    dataset["urine_protein"] = dataset["urine_protein"].astype(int).astype(object)
    dataset["hear_left"] = dataset["hear_left"].astype(int).astype(object)
    dataset["hear_right"] = dataset["hear_right"].astype(int).astype(object)
    return dataset


@st.cache_resource
def load_model():
    full_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    st.info(f"Ielādē modeli: {full_uri}")
    model = mlflow.sklearn.load_model(full_uri)
    st.success("Modelis ielādēts")
    return model


def main():
    st.title("Alkohola lietošanas statusa prognoze")
    uploaded_file = st.file_uploader("Augšupielādējiet CSV", type="csv")

    if uploaded_file is not None:
        try:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            data_df = pd.read_csv(stringio)
            st.subheader("Datu priekšskatījums")
            st.dataframe(data_df.head(), use_container_width=True)

            dataset = preprocess_input(data_df)
            model = load_model()
            predictions = model.predict(dataset.drop(columns=[TARGET]))

            results_df = pd.DataFrame(
                {"Record Index": dataset.index, "Predicted DRK_YN": predictions}
            )
            st.subheader("Prognozes")
            st.dataframe(results_df, use_container_width=True)

        except Exception as e:
            st.error(f"Kļūda: {e}")


if __name__ == "__main__":
    main()
