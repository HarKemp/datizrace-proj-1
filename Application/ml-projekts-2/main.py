import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Sklearn imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.base import BaseEstimator, TransformerMixin

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# CONFIG
warnings.filterwarnings("ignore")
MLFLOW_TRACKING_URI = "http://172.16.238.23:5000"
EXPERIMENT_NAME = "Hotel-Cancelation-Pred"
RANDOM_SEED = 80085

# CLIPPER
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

# SETUP MLFLOW
def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    existing_exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if existing_exp is None:
        try:
            mlflow.create_experiment(EXPERIMENT_NAME)
        except Exception as e:
            print(f"Note: Experiment already existed or could not be created: {e}")
    
    mlflow.set_experiment(EXPERIMENT_NAME)

# PREPROCESS
def load_and_preprocess_data():

    possible_paths = [
        'data/Hotel Reservations.csv',       # Ja palaiž uz docker
        '../Data/Hotel Reservations.csv',    # Ja palaiž notebook
    ]
    
    filename = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if not filename:
        raise FileNotFoundError(f"Could not find 'Hotel Reservations.csv'. Checked paths: {possible_paths}")
    
    df = pd.read_csv(filename)

    # Definējam mērķa mainīgo (y)
    # Nepieciešams prognozēt 'booking_status'
    if 'Booking_ID' in df.columns:
        df = df.drop(['Booking_ID'], axis=1)

    # Definējam atribūtus (X)
    # Dati, ko izmantosim prognozēšanai.
    # Izmetam 'Booking_ID', jo tas ir tikai identifikators un nav noderīgs prognozēšanai.
    # Izmetam pašu 'booking_status', jo tas ir mūsu mērķia atribūts.
    X = df.drop(['booking_status'], axis=1)
    y = df['booking_status']

    # Canceled / Not_Canceled -> 0 / 1
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Datu sadalīšana
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, 
        random_state=RANDOM_SEED, 
        stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, le

# PIPELINE
def get_processing_pipeline():
    # Skaitliskie atribūti, kuriem izmanto StandardScaler
    numeric_features = [
        'no_of_weekend_nights', 'no_of_week_nights', 'lead_time', 
        'no_of_adults', 'no_of_children', 'no_of_special_requests', 
        'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled'
    ]
    avg_price_feature = ['avg_price_per_room']
    
    # Kategoriskie atribūti, kuriem izmanto OneHotEncoder
    categorical_features = [
        'type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'arrival_month', 'arrival_year'
    ]
    
    # Binārie atribūti, kuriem neizmanto nekādu pārveidošanu
    passthrough_features = ['required_car_parking_space', 'repeated_guest']

    # Transformatoru piemērošana katrai grupai

    # Skaitliskais transformators:
    numeric_transformer = Pipeline(steps=[
        ('scaler', RobustScaler())
    ])

    avg_price_transformer = Pipeline(steps=[
        ('clip', QuantileClipper(0.1, 0.99)),
        ('scaler', StandardScaler())
    ])

    # Kategoriskais transformators:
    # handle_unknown='ignore' nodrošina, ka modelis neizmetīs kļūdu, 
    # ja testēšanas datos parādīsies kategorija, kas nebija apmācības datos.
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Transformatoru apvienošana
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('price', avg_price_transformer, avg_price_feature),
            ('cat', categorical_transformer, categorical_features),
            ('pass', 'passthrough', passthrough_features)
        ],
        remainder='drop' #  Visas pārējās kolonnas (t.i., 'arrival_date') tiks atmestas
    )
    return preprocessor


# MODEL PARAMETERS
def get_models_and_params():
    models = []

    # Modelis 1: Loģistiskā Regresija
    # class_weight='balanced' svarīgs, jo mērogs ir atšķirīgs un dati ir nelīdzsvaroti
    lr = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, class_weight='balanced')
    grid_lr = {
        'model__C': [0.1, 1.0, 10],
        'model__solver': ['liblinear', 'saga']
    }
    models.append(('Logistic_Regression', lr, grid_lr))

    # Modelis 2: Lēmumu Koks
    dt = DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight='balanced')
    grid_dt = {
        'model__max_depth': [5, 10, 20, None],
        'model__min_samples_leaf': [1, 2, 5]
    }
    models.append(('Decision_Tree', dt, grid_dt))

    # Modelis 3: Gadījuma Mežs
    rf = RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced')
    grid_rf = {
        'model__n_estimators': [100, 150],
        'model__max_depth': [10, 20],
        'model__min_samples_leaf': [1, 3]
    }
    models.append(('Random_Forest', rf, grid_rf))

    # Modelis 4: XGBoost
    xgb_clf = xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss')
    grid_xgb = {
        'model__n_estimators': [100, 150],
        'model__max_depth': [5, 10],
        'model__learning_rate': [0.1, 0.2],
    }
    models.append(('XGBoost', xgb_clf, grid_xgb))

    return models

# GENERATE PLOTS
def generate_plots(model, X_test, y_test, model_name):
    from sklearn.metrics import roc_curve, auc, confusion_matrix

    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")

    # 1. ROC Curve
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = model.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {model_name}')
    plt.legend(loc="lower right")
    
    roc_path = f"artifacts/{model_name}_ROC.png"
    plt.savefig(roc_path)
    plt.close()

    # 2. Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_path = f"artifacts/{model_name}_CM.png"
    plt.savefig(cm_path)
    plt.close()

    return roc_path, cm_path, roc_auc

# MAIN
def main():
    setup_mlflow()
    print("Loading data")
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data()
    
    preprocessor = get_processing_pipeline()
    
    models = get_models_and_params()
    
    best_overall_model_name = None
    best_overall_accuracy = 0
    best_run_id = None
    
    print("=== Modeļu apmācība un hiperparametru pielāgošana ===")
    
    for name, model, param_grid in models:
        print(f"\nUzsāk apmācību modelim: {name}")
        
        # Create full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Grid Search
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=10,
            n_jobs=-1,
            scoring='f1_weighted',
            verbose=0
        )
        
        # Start MLflow Run
        with mlflow.start_run(run_name=name) as run:
            # Fit
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Plots
            roc_path, cm_path, roc_auc = generate_plots(best_model, X_test, y_test, name)
            
            # Logging to MLflow
            mlflow.set_tag("model_type", name)
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_weighted", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            
            mlflow.log_artifact(roc_path)
            mlflow.log_artifact(cm_path)
            
            # Log Model (sklearn)
            input_example = X_train.iloc[:5]
            mlflow.sklearn.log_model(
                best_model, 
                "model",
                input_example=input_example
            )
            
            print(f"  - {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}")
            
            # Champion izvēles parametri
            if accuracy > best_overall_accuracy:
                best_overall_accuracy = accuracy
                best_overall_model_name = name
                best_run_id = run.info.run_id

    print("\n" + "="*30)
    print(f"CHAMPION MODEL: {best_overall_model_name} with Accuracy: {best_overall_accuracy:.4f}")
    print("="*30)
    
    # Reģistrē champion modeli
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        registered_model_name = "Hotel_Cancellation_Model"
        
        try:
            mv = mlflow.register_model(model_uri, registered_model_name)
            
            # Izveido alias
            client = MlflowClient()
            client.set_registered_model_alias(
                name=registered_model_name,
                alias="champion",
                version=mv.version
            )
            print(f"Successfully registered model version {mv.version} as 'champion'.")
            
        except Exception as e:
            print(f"Error registering model: {e}")

if __name__ == "__main__":
    main()