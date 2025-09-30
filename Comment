# Análise Comparativa de Estratégias de Validação Cruzada

Este notebook tem como objetivo comparar o impacto de duas diferentes estratégias de validação cruzada no desempenho de um classificador SVM (Support Vector Machine). A análise é realizada no dataset *Pima Indians Diabetes*, um problema clássico de classificação biomédica.

As estratégias comparadas são:

1.  **K-Fold (10 splits):** Uma abordagem comum.
2.  **Repeated Stratified K-Fold (20x2):** Uma abordagem mais robusta, que estratifica os dados e repete o processo para garantir maior estabilidade nos resultados.

## FASE 1: Configuração e Carga de Dados

Nesta primeira fase, preparamos todo o ambiente para a nossa análise. Isso inclui importar as bibliotecas necessárias, definir as constantes globais e carregar os dados, aplicando um pré-processamento crucial.

### 1.1 - Importações e Configurações Globais

Começamos importando todas as ferramentas que vamos utilizar. Em seguida, definimos nossas configurações globais, como a URL do dataset, o nome das colunas, os hiperparâmetros que o `GridSearchCV` irá testar e, mais importante, as duas estratégias de validação cruzada que serão o foco do nosso estudo.

Note que o `PARAM_GRID` é uma lista de dicionários. Isso permite testar o `kernel='linear'` sem o parâmetro `gamma` (que não é utilizado por ele) e o `kernel='rbf'` com seus respectivos valores de `gamma`, tornando a busca mais eficiente.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    KFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from typing import Dict, Any

# --- Configurações Globais ---
DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMN_NAMES = [
    'Nº de Gestações', 'Concentração de Glicose', 'Pressão Arterial (mmHg)',
    'Espessura da Pele (mm)', 'Nível de Insulina (mu U/ml)', 'Índice de Massa Corporal',
    'Histórico Familiar de Diabetes', 'Idade (anos)', 'Classe'
]
TARGET_COLUMN = 'Classe'

# Grid de parâmetros otimizado para o GridSearchCV
PARAM_GRID = [
    {
        'svm__C': [0.01, 0.1, 1, 10, 100],
        'svm__kernel': ['linear']
    },
    {
        'svm__C': [0.01, 0.1, 1, 10, 100],
        'svm__kernel': ['rbf'],
        'svm__gamma': ['scale', 0.01, 0.001]
    }
]

# Dicionário com as estratégias de validação a serem comparadas
CV_STRATEGIES = {
    "K-Fold (10 splits)": KFold(n_splits=10, shuffle=True, random_state=42),
    "Repeated Stratified K-Fold (20x2)": RepeatedStratifiedKFold(n_splits=20, n_repeats=2, random_state=42)
}
```

### 1.2 - Carga e Pré-processamento dos Dados

Aqui, carregamos o dataset usando o `pandas`. O passo mais importante nesta etapa é o **tratamento de valores ausentes**. No dataset Pima, valores ausentes são representados pelo número `0` em colunas onde isso é biologicamente impossível (ex: uma pessoa não pode ter 'Concentração de Glicose' igual a zero).

Nós substituímos esses zeros por `np.nan` (Not a Number), que é o marcador padrão para dados ausentes em Python. Isso permite que a etapa de imputação (`SimpleImputer`) na FASE 3 funcione corretamente.

```python
print_header("FASE 1: CONFIGURAÇÃO E CARGA DE DADOS")
try:
    df = pd.read_csv(DATASET_URL, header=None, names=COLUMN_NAMES)
    
    # --- Tratamento Crítico de Valores Ausentes ---
    print("\n -> Verificando e tratando valores ausentes (representados por '0')...")
    cols_com_zeros_ausentes = [
        'Concentração de Glicose', 'Pressão Arterial (mmHg)',
        'Espessura da Pele (mm)', 'Nível de Insulina (mu U/ml)', 'Índice de Massa Corporal'
    ]
    df[cols_com_zeros_ausentes] = df[cols_com_zeros_ausentes].replace(0, np.nan)
    print(f" -> Contagem de valores nulos após substituição:\n{df.isnull().sum()}")

    # Separação das features (X) e do alvo (y)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

except Exception as e:
    print(f" -> Falha ao carregar o dataset: {e}")
    X, y = None, None
```

## FASE 2: Análise Exploratória de Dados (EDA)

Antes de treinar qualquer modelo, é fundamental entender os dados. Esta fase gera duas visualizações importantes:

1.  **Histogramas:** Mostram a distribuição de cada variável, separada por classe (diabéticos e não diabéticos).
2.  **Heatmap de Correlação:** Mostra a relação linear entre as variáveis. É útil para identificar preditores fortes e evitar multicolinearidade.

<!-- end list -->

```python
def plot_eda_graphs(X: pd.DataFrame, y: pd.Series) -> None:
    print_header("FASE 2: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")

    # --- Heatmap de Correlação ---
    print("\n--- Heatmap de Correlação entre as Features ---")
    plt.figure(figsize=(10, 8))
    sns.heatmap(X.corr(), annot=True, cmap='viridis', fmt='.2f')
    plt.title('Correlação entre as Variáveis Preditivas')
    plt.show()

# Chamada da função
plot_eda_graphs(X, y)
```

## FASE 3: Metodologia e Execução da Análise Comparativa

Esta é a fase central do estudo. Aqui, construímos nosso pipeline de machine learning e o executamos para cada uma das estratégias de validação cruzada.

### 3.1 - Construção do Pipeline

Usamos um `Pipeline` do Scikit-learn para encadear as etapas de pré-processamento e modelagem. Isso é crucial para **evitar vazamento de dados (data leakage)**, pois garante que a imputação de dados ausentes e a padronização sejam calculadas usando apenas os dados de treino de cada *fold* da validação cruzada.

Nosso pipeline executa 3 passos:

1.  `SimpleImputer`: Preenche os valores `NaN` com a média da coluna.
2.  `StandardScaler`: Padroniza as features (média 0, desvio padrão 1).
3.  `SVC`: O classificador Support Vector Machine.

<!-- end list -->

```python
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=42, probability=True))
])
```

### 3.2 - Execução e Geração de Resultados

Iteramos sobre cada estratégia de validação (`KFold` e `RepeatedStratifiedKFold`). Em cada iteração:

1.  Executamos o `GridSearchCV` para encontrar a melhor combinação de hiperparâmetros.
2.  Extraímos a acurácia média e o desvio padrão.
3.  Geramos uma **matriz de confusão robusta** usando `cross_val_predict`.

**Ponto Chave:** `cross_val_predict` não funciona com validadores "repetidos". Para contornar isso, verificamos se a estratégia é `RepeatedStratifiedKFold`. Se for, usamos um `StratifiedKFold` simples (que é uma partição) apenas para gerar as predições da matriz, garantindo uma avaliação justa dos erros sem causar falhas no código.

```python
def run_comparative_analysis(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    # ... (início da função)
    
    for name, cv_strategy in CV_STRATEGIES.items():
        # ... (código do GridSearchCV)

        # Lógica para gerar predições para a Matriz de Confusão
        if isinstance(cv_strategy, RepeatedStratifiedKFold):
            # Obtém o n_splits de forma segura
            n_splits = cv_strategy.get_n_splits()
            # Usa um validador de partição para o cross_val_predict
            cv_for_predict = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv_for_predict = cv_strategy

        y_pred = cross_val_predict(best_pipeline, X, y, cv=cv_for_predict, n_jobs=-1)
        cm = confusion_matrix(y, y_pred)

        # ... (armazenamento de resultados e plot da matriz)
    return results_summary

# Chamada da função
final_results = run_comparative_analysis(X, y)
```

## FASE 4: Análise Comparativa Final dos Resultados

Com os resultados em mãos, esta fase foca em visualizá-los de forma clara para extrair conclusões.

1.  **Gráfico de Barras:** Compara diretamente a acurácia média e o desvio padrão de cada estratégia.
2.  **Boxplot:** Uma visualização mais poderosa que mostra a **distribuição completa dos scores** de acurácia em todos os folds. Ele nos permite avaliar a **estabilidade** e a **consistência** de cada método, revelando que a acurácia média nem sempre conta a história toda.

<!-- end list -->

```python
def plot_final_comparison(results: Dict[str, Any]) -> None:
    print_header("FASE 4: ANÁLISE COMPARATIVA FINAL DOS RESULTADOS")

    # --- Boxplot da Distribuição dos Scores ---
    all_scores_data = []
    for name, result in results.items():
        cv_results_df = result['cv_results']
        # Filtra a linha correspondente aos melhores parâmetros
        best_param_mask = cv_results_df['params'].apply(lambda d: d == result['melhores_parametros'])
        
        # Pega as colunas de score de cada split
        split_score_columns = [col for col in cv_results_df.columns if 'split' in col and 'test_score' in col]
        
        # Extrai os scores e adiciona à lista para o plot
        scores = cv_results_df.loc[best_param_mask, split_score_columns].values.flatten()
        for score in scores:
            all_scores_data.append({'Estratégia': name, 'Acurácia': score})

    scores_df = pd.DataFrame(all_scores_data)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Estratégia', y='Acurácia', data=scores_df, palette='viridis')
    sns.stripplot(x='Estratégia', y='Acurácia', data=scores_df, color='black', alpha=0.3, jitter=0.2)
    plt.title('Distribuição dos Scores de Acurácia nos Folds')
    plt.show()

# Chamada da função
plot_final_comparison(final_results)
```
