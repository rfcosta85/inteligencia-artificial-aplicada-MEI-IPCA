!pip install supabase
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import numpy as np
import random
from supabase import create_client
from google.colab import drive
from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, brier_score_loss, \
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
from mlxtend.plotting import plot_decision_regions
from tabulate import tabulate

drive.mount('/content/drive')

csv_path = '/content/drive/MyDrive/equipment_anomaly_data.csv'

supabaseKey = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlqa2JmamRnZWpxd3dubm9wc2JtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzYwMDg1NDksImV4cCI6MjA1MTU4NDU0OX0.YcXQIL4CUncd6lUX6CvPDDFXhQcPFOqcaw4MdSTd9M8"
supabaseUrl = 'https://yjkbfjdgejqwwnnopsbm.supabase.co'
supabase = create_client(supabaseUrl, supabaseKey)

df = pd.read_csv(csv_path)

# Realiza o mapeamento da coluna equipment
equipment_number = {'Turbine': 0, 'Compressor': 1, 'Pump': 2}
df['equipment'] = df['equipment'].replace(equipment_number)

# Transforma a coluna faulty de float para int
df['faulty'] = df['faulty'].round().astype(int)

# Transforma a coluna location de string para int
encoder = LabelEncoder()
df['location'] = encoder.fit_transform(df['location'])

# Visualizar as primeiras linhas do DataFrame
print(df.head())

# Supondo que a coluna "alvo" seja o rótulo e o resto são features
X = df.drop(['faulty'], axis=1)  # Remove a coluna 'faulty', 'location'
y = df['faulty'] # Seleciona a coluna 'faulty'

# Verificar as dimensões
print("Formato de X:", X.shape)
print("Formato de y:", y.shape)


# Função genérica para calcular métricas
def calculate_metrics(y_test, predict):
    tn, fp, fn, tp = confusion_matrix(y_test, predict).ravel()
    return {
        "accuracy": accuracy_score(y_test, predict),
        "precision": precision_score(y_test, predict),
        "recall": recall_score(y_test, predict),
        "f1": f1_score(y_test, predict),
        "roc_auc": roc_auc_score(y_test, predict),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "true_negative_rate": tn / (tn + fp),
            "false_positive_rate": fp / (tn + fp),
            "false_negative_rate": fn / (fn + tp),
            "true_positive_rate": tp / (fn + tp)
        },
        "specific": None
    }

def plot_svm_decision_boundary(model, X_train, y_train):
    if X_train.shape[1] < 2:
        print("Não há features suficientes para plotar a fronteira de decisão.")
        return

    # Pegamos apenas as duas primeiras colunas para visualização
    feature_1 = X_train.columns[0]
    feature_2 = X_train.columns[1]

    X_subset = X_train[[feature_1, feature_2]]

    # Pegamos os coeficientes do modelo SVM
    w = model.coef_[0]
    w0 = model.intercept_[0]

    # Criamos os pontos para a linha de decisão
    x = np.linspace(X_subset.iloc[:, 0].min(), X_subset.iloc[:, 0].max(), 100)
    y = (-w[0] * x - w0) / w[1]

    # Plotamos os pontos e a fronteira de decisão
    plt.figure(figsize=(8, 6))
    plt.scatter(X_subset.iloc[:, 0], X_subset.iloc[:, 1], c=y_train, s=60, cmap='coolwarm', edgecolors='k')
    plt.plot(x, y, 'r', label="Fronteira de decisão")
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.title("SVM - Fronteira de Decisão")
    plt.legend()
    plt.show()

# Recriar o modelo
def recreate_model(model_name, test_size, random_state):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state= random_state)
  model = DecisionTreeClassifier()
  match model_name:
    case 'DecisionTreeClassifier':
      model = DecisionTreeClassifier()
    case  'SVC':
      model = SVC
    case 'MLPClassifier':
      model = MLPClassifier()
    case 'LogisticRegression':
      model = LogisticRegression()
    case 'KNeighborsClassifier':
      model = KNeighborsClassifier()
    case 'RandomForestClassifier':
      model = RandomForestClassifier()
  model.fit(X_train, y_train)
  return model

# Função para treinar e avaliar um modelo
def train_and_evaluate_model(model, X, y, test_size, random_state, results):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state= random_state)
    model.fit(X_train, y_train)
    results.append({
        "model": model.__class__.__name__,
        "metrics": calculate_metrics(y_test, model.predict(X_test)),
        "hyperparameters": {
            "test_size": test_size,
            "random_state": random_state
        }
    })
    import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Modificando a função para incluir o gráfico SOMENTE se for test_size == 0.2
def train_and_evaluate_model(model, X, y, test_size, random_state, results):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)

    # Salva os resultados
    results.append({
        "model": model.__class__.__name__,
        "metrics": calculate_metrics(y_test, model.predict(X_test)),
        "hyperparameters": {
            "test_size": test_size,
            "random_state": random_state
        }
    })
   
lock = threading.Lock()
max_threads = 25
results = []
test_sizes = [0.2, 0.3, 0.5, 0.7]
random_states = [1, 42, 101]
models = [
    DecisionTreeClassifier,
    SVC,
    MLPClassifier,
    LogisticRegression,
    KNeighborsClassifier,
    RandomForestClassifier
]

with ThreadPoolExecutor(max_workers= max_threads) as executor:
  futures = []
  for random_state in random_states:
    for test_size in test_sizes:
      for model in models:
        if model == KNeighborsClassifier:
          futures.append(executor.submit(train_and_evaluate_model, model(), X, y, test_size, random_state, results))
        else:
          futures.append(executor.submit(train_and_evaluate_model, model(random_state= random_state), X, y, test_size, random_state, results))
  for future in as_completed(futures):
    try:
      future.result()
    except Exception as e:
          print(f"Thread error: {e}")

# Exibindo os resultados em formato de tabela
columns = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "specific",
    "cm_true_negative",
    "cm_false_positive",
    "cm_false_negative",
    "cm_true_positive",
    "cm_true_negative_rate",
    "cm_true_positive_rate",
    "cm_false_negative_rate",
    "cm_false_positive_rate",
    "test_size",
    "random_state"
    ]

results_df = pd.DataFrame(columns=columns)
for row in results:
  model = {
      "model": row["model"],
      "accuracy": row["metrics"]["accuracy"],
      "precision": row["metrics"]["precision"],
      "recall": row["metrics"]["recall"],
      "f1": row["metrics"]["f1"],
      "roc_auc": row["metrics"]["roc_auc"],
      "specific": row["metrics"]["specific"],
      "cm_true_negative": row["metrics"]["confusion_matrix"]["true_negative"],
      "cm_false_positive": row["metrics"]["confusion_matrix"]["false_positive"],
      "cm_false_negative": row["metrics"]["confusion_matrix"]["false_negative"],
      "cm_true_positive": row["metrics"]["confusion_matrix"]["true_positive"],
      "cm_true_negative_rate": row["metrics"]["confusion_matrix"]["true_negative_rate"],
      "cm_true_positive_rate": row["metrics"]["confusion_matrix"]["true_positive_rate"],
      "cm_false_negative_rate": row["metrics"]["confusion_matrix"]["false_negative_rate"],
      "cm_false_positive_rate": row["metrics"]["confusion_matrix"]["false_negative_rate"],
      "test_size": row["hyperparameters"]["test_size"],
      "random_state": row["hyperparameters"]["random_state"]
  }
  results_df.loc[len(results_df)] = model

  model_serializable = {key: int(value) if isinstance(value, (np.int64, np.int32)) else float(value) if isinstance(value, (np.float64, np.float32)) else value for key, value in model.items()}
  supabase.table("models").insert(model_serializable).execute()

file_name = 'models-results-with-random-forest.csv'
results_df.to_csv(file_name, index=False, encoding='utf-8')

# Escolher o melhor modelo
best_models = supabase.table('models') \
    .select('*') \
    .order('recall', desc=True) \
    .order('f1', desc=True) \
    .order('roc_auc', desc=True) \
    .order('cm_true_negative_rate', desc=True) \
    .limit(3) \
    .execute()
best_model = best_models.data[0]
# Imprimir os resultados do melhor modelo
print("Top 3 Models:")
for m in best_models.data:
  print(m)

row_select_number = random.randint(0, len(df)-1)
print('Linha analisada', row_select_number)
row_selected =  df.iloc[row_select_number:row_select_number + 1]
machine_real_state = row_selected["faulty"].values[0]
caseToPredict = row_selected.drop(columns=["faulty"])
betterModel = recreate_model(best_model["model"], best_model["test_size"], best_model["random_state"])
print(f'Real: A máquina analisada, {"está com defeito" if machine_real_state == 1 else "não está com defeito"}')
print(f'Prediction: A máquina analisada, {"está com defeito" if betterModel.predict(caseToPredict)[0] == 1 else "não está com defeito"}')

# Selecionar as métricas principais para comparação
metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Plotando um gráfico de barras
plt.figure(figsize=(10,6))
sns.barplot(x="model", y="recall", data=results_df.sort_values("recall", ascending=False))
plt.xticks(rotation=45)
plt.title("Modelos com Melhor Recall")
plt.show()

# Melhores accuracy
plt.figure(figsize=(10,6))
sns.barplot(x="model", y="accuracy", data=results_df.sort_values("accuracy", ascending=False))
plt.xticks(rotation=45)
plt.title("Modelos com melhor accuraccy")

# Melhores precision
plt.figure(figsize=(10,6))
sns.barplot(x="model", y="precision", data=results_df.sort_values("precision", ascending=False))
plt.xticks(rotation=45)
plt.title("Modelos com melhor precision")

# Melhores F1
plt.figure(figsize=(10,6))
sns.barplot(x="model", y="f1", data=results_df.sort_values("f1", ascending=False))
plt.xticks(rotation=45)
plt.title("Modelos com melhor F1")

# Melhores roc_auc
plt.figure(figsize=(10,6))
sns.barplot(x="model", y="roc_auc", data=results_df.sort_values("roc_auc", ascending=False))
plt.xticks(rotation=45)
plt.title("Modelos com melhor roc_auc")
