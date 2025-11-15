import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import mlflow
import os
import warnings
warnings.filterwarnings("ignore")  

experiment_name = 'Aizdevumu novērtēšana'  # Eksperimenta nosaukums, kurā tiks glabāti mūsu palaidieni (RUN)

# Ielādējam datu kopu
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data/train.csv')
dataset = pd.read_csv(filename)
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

# Pieņemot ka datu kopā var būt nepiederošie dati (Outliers),
# atmetam vērtības, kas ir pirmajos 5% un pēdējos 5% pēc vērtības
dataset[numerical_cols] = dataset[numerical_cols].apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))

# Skaitlisko kolonnu normalizācija un kopējā ienakuma aprēķins
dataset['LoanAmount'] = np.log(dataset['LoanAmount']).copy()
dataset['TotalIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncome'] = np.log(dataset['TotalIncome']).copy()

# Nevajadzīgo kolonnu nodzēšana
dataset = dataset.drop(columns=['ApplicantIncome','CoapplicantIncome'])

# Kategoriju atribūtu vērtību iekodēšana ar skaitļiem. Tiek pielietots LabelEncoder, kurš to dara automātiski.
for col in categorical_cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])

# Mērķā atribūta vērtību iekodēšana
dataset['Loan_Status'] = le.fit_transform(dataset['Loan_Status'])

# Datu kopas sadalīšana apmācības un testēšanas kopās.
# Vispirms atdalam aprakstošus atribūtus un mērķa atribūtu
X = dataset.drop(columns=['Loan_Status', 'Loan_ID'])
y = dataset.Loan_Status
RANDOM_SEED = 6
# Tad sadalam apmācības un testēšanas kopās
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.3, random_state = RANDOM_SEED)

# Apmācām modeļus
# Vispirms RandomForest
rf = RandomForestClassifier(random_state=RANDOM_SEED)
# Definējam parametrus, kuri tiks padoti modeļa apmācības laikā, un norādām vērtības, kuras mēs vēlamies pārbaudīt.
# Apmācības laikā notiks visu norādīto parametru vērtību kombinācijas.
# Parametru nosaukumiem ir jābūt tādiem pašiem, kā norādīts algoritma API aprakstā.
param_grid_forest = {
    'n_estimators': [25,50],
    'max_depth': [10,20]
}

# Izveidojam objektu parametru kombināciju novērtēšanai
grid_forest = GridSearchCV(
        estimator=rf,
        param_grid=param_grid_forest, 
        cv=5, 
        n_jobs=-1, 
        scoring='accuracy',
        verbose=0
    )
# Palaižam apmācības un novērtēšanas procesu
model_forest = grid_forest.fit(X_train, y_train)

# Pārbaudam vai mūsu eksperiments eksistē un ja ne, tad to izveidojam
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Funkcija efektivitātes rādītāju (metriku) novērtēšanai un grafika izveidošanai
def eval_metrics(actual, pred):
    # Novērtējam metrikas
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    # Izveidojam grafikus AUC attēlošanai
    plt.figure(figsize=(8,8))   # Grafika izmērs collās
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f'%auc)    # Žīmējam līniju
    plt.plot([0,1],[0,1], 'r--')    # Novilkam diagonāles līniju
    plt.xlim([-0.1, 1.1])       # Definējam X ass limīrtu nedaudz plašāk, lai grafiks izskatītos labāk
    plt.ylim([-0.1, 1.1])       # Definējam Y ass limīrtu nedaudz plašāk, lai grafiks izskatītos labāk
    plt.xlabel('False Positive Rate', size=14)  # X ass nosaukums
    plt.ylabel('True Positive Rate', size=14)   # Y ass nosaukums
    plt.legend(loc='lower right')   # Leģendas izveidošana un novietošana labajā apakšējā stūrī
    # Saglabājam grafiku artifaktu mapē
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/ROC_curve.png")
    # Iznicinām grafika objektu, lai tur nekas vairs netiktu zīmēts
    plt.close()
    # Atgriežam metriku vērtības noteiktajā secībā
    return(accuracy, f1, auc)

# Funkcija eksperimentu informācijas ierakstīšanai žurnālā
def mlflow_logging(model, X, y, name):
    # Sākam palaidumu ar attiecīgo nosaukumu
     with mlflow.start_run(run_name=name) as run:
        # Fiksējam palaiduma numuru kā tag'u
        run_id = run.info.run_id
        mlflow.set_tag("run_id", str(run_id))    
        # Prognozējam vērtības norādītajai (testēšanas) kopai
        pred = model.predict(X)
        # Aprēķinam metriku vērtības
        (accuracy, f1, auc) = eval_metrics(y, pred)
        # Fiksējam žurnālā labākos modeļa parametrus
        mlflow.log_params(model.best_params_)
        # Fiksējam žurnālā metriku vērtības
        mlflow.log_metric("Mean CV score", model.best_score_)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1-score", f1)
        mlflow.log_metric("AUC", auc)

        # Fiksējam artifaktu un modeli
        mlflow.log_artifact("artifacts/ROC_curve.png", name)
        mlflow.sklearn.log_model(model, 
                                 name = name,
                                 input_example=X.iloc[:2].to_dict(orient='records'))

# Palaižam modeļu testēšanu un fiksāciju žurnālā.
mlflow_logging(model_forest, X_test, y_test, "RandomForestClassifier")

