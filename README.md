-----

## FASE 1: Configuração e Carga de Dados

Tudo começa com a preparação do ambiente. Esta fase é responsável por importar as bibliotecas, definir as constantes e carregar o dataset, realizando um pré-processamento essencial.

```python
# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer

# Definição das estratégias de validação cruzada
CV_STRATEGIES = {
    "K-Fold (10 splits)": KFold(n_splits=10, shuffle=True, random_state=42),
    "Repeated Stratified K-Fold (20x2)": RepeatedStratifiedKFold(n_splits=20, n_repeats=2, random_state=42)
}

# Carregamento e tratamento inicial dos dados
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", header=None, names=COLUMN_NAMES)
cols_com_zeros_ausentes = [
    'Concentração de Glicose', 'Pressão Arterial (mmHg)',
    'Espessura da Pele (mm)', 'Nível de Insulina (mu U/ml)', 'Índice de Massa Corporal'
]
df[cols_com_zeros_ausentes] = df[cols_com_zeros_ausentes].replace(0, np.nan)
X = df.drop('Classe', axis=1)
y = df['Classe']
```

**Resumo da Fase 1:**

  - **Configuração:** Todas as ferramentas são importadas e as duas estratégias de validação (`KFold` e `RepeatedStratifiedKFold`) são definidas em um dicionário para facilitar a iteração posterior.
  - **Carga de Dados:** O dataset Pima Indians Diabetes é carregado via URL.
  - **Limpeza de Dados:** Um passo crucial é realizado aqui. Em algumas colunas, como 'Concentração de Glicose', o valor `0` é fisicamente impossível e, na verdade, representa um dado ausente. O código substitui esses zeros por `np.nan` (Not a Number), que é o marcador correto para valores faltantes. Isso prepara o terreno para que o `SimpleImputer` funcione corretamente na FASE 3.

## FASE 2: Análise Exploratória de Dados (EDA)

Antes de treinar o modelo, é vital entender a natureza dos dados. A função `plot_eda_graphs` faz exatamente isso, gerando histogramas e um heatmap.

```python
def plot_eda_graphs(X: pd.DataFrame, y: pd.Series) -> None:
    # ... (código dos histogramas) ...

    # --- Heatmap de Correlação entre as Features ---
    print("\n--- Heatmap de Correlação entre as Features ---")
    plt.figure(figsize=(10, 8))
    sns.heatmap(X.corr(), annot=True, cmap='viridis', fmt='.2f')
    plt.title('Correlação entre as Variáveis Preditivas')
    plt.show()
```

### O que é o Heatmap e como ele é usado aqui?

O **Heatmap (Mapa de Calor)** é uma técnica de visualização de dados que representa a magnitude de um fenômeno em cores. Em ciência de dados, ele é frequentemente usado para visualizar uma matriz de correlação, como acontece no seu código.

**Código base:**

  - **Função:** O heatmap é usado para visualizar a matriz de correlação (`X.corr()`) entre todas as variáveis preditoras.
  - **Interpretação:**
      - Cada célula do mapa mostra o coeficiente de correlação entre duas variáveis (um número de -1 a 1).
      - **Cores quentes/claras** (amarelo no seu gráfico) indicam uma correlação positiva forte. Por exemplo, a correlação entre 'Idade' e 'Nº de Gestações' é a mais alta.
      - **Cores frias/escuras** (roxo/azul) indicam uma correlação fraca ou negativa.
  - **Objetivo:** O objetivo aqui é entender as relações entre as variáveis antes da modelagem. Uma correlação muito alta entre duas variáveis preditoras (ex: \> 0.9) poderia indicar multicolinearidade, um problema onde as variáveis são redundantes. Além disso, observar a correlação das features com a variável alvo (neste caso, 'Classe', que não está no heatmap de `X.corr()`, mas estaria se o fizéssemos com o dataframe completo) pode dar pistas sobre quais são os preditores mais importantes.

> **Nota:** Sobre o uso do heatmap para visualizar a matriz de confusão, é importante notar que **neste script específico**, o heatmap é usado para a **matriz de correlação**. A matriz de confusão é visualizada com a função `ConfusionMatrixDisplay`, que internamente também gera uma visualização colorida similar a um heatmap.

## FASE 3: Execução da Análise Comparativa

Esta é a parte central do projeto, onde o modelo é treinado e avaliado.

### O Pipeline

O código utiliza um `Pipeline` para encadear os passos de pré-processamento e modelagem. Isso é uma excelente prática, pois previne o vazamento de dados (*data leakage*), garantindo que a média para imputação e os parâmetros de padronização sejam aprendidos apenas com os dados de treino de cada fold.

```python
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=42, probability=True))
])
```

### A Matriz de Confusão

Após o `GridSearchCV` encontrar os melhores parâmetros, o código gera as predições e calcula a matriz de confusão.

```python
# Gera as predições de forma robusta
y_pred = cross_val_predict(best_pipeline, X, y, cv=cv_for_predict, n_jobs=-1)

# Calcula a matriz de confusão
cm = confusion_matrix(y, y_pred)

# Plota a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Não Diabético', 'Diabético'])
disp.plot(cmap=plt.cm.Blues, ax=ax)
```

**O que é a Matriz de Confusão?**

A **Matriz de Confusão** é uma tabela que descreve o desempenho de um modelo de classificação. Ela compara os valores reais com as predições do modelo, permitindo uma análise detalhada dos tipos de erro. A matriz é dividida em quatro quadrantes:

  - **Verdadeiros Positivos (VP):** Canto inferior direito. Previsão: Diabético, Real: Diabético. (O modelo acertou o positivo).
  - **Verdadeiros Negativos (VN):** Canto superior esquerdo. Previsão: Não Diabético, Real: Não Diabético. (O modelo acertou o negativo).
  - **Falsos Positivos (FP):** Canto superior direito. Previsão: Diabético, Real: Não Diabético. (Erro Tipo I: alarme falso).
  - **Falsos Negativos (FN):** Canto inferior esquerdo. Previsão: Não Diabético, Real: Diabético. (Erro Tipo II: o erro mais perigoso neste contexto clínico).

**Como ela é gerada no código?**
Ela é gerada pela função `confusion_matrix(y_true, y_pred)`, que recebe os valores reais (`y`) e as predições (`y_pred`) feitas pelo `cross_val_predict`.

**O que os resultados significam?**
A matriz de confusão vai além da acurácia. Ela nos mostra *onde* o modelo está errando. Em um problema médico, minimizar os **Falsos Negativos** é geralmente a maior prioridade. Ao comparar as matrizes de confusão geradas pelo `KFold` e pelo `RepeatedStratifiedKFold`, você pode analisar qual estratégia leva a um modelo com menos erros perigosos.

## FASE 4: Análise Comparativa Final dos Resultados

Esta fase visualiza os resultados agregados para permitir uma conclusão robusta. Além do gráfico de barras, ela utiliza um boxplot.

```python
# ... (código para preparar os dados para o boxplot) ...

# Cria o boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Estratégia', y='Acurácia', data=scores_df, palette='viridis')
sns.stripplot(x='Estratégia', y='Acurácia', data=scores_df, color='black', alpha=0.3, jitter=0.2)
plt.title('Distribuição dos Scores de Acurácia nos Folds')
plt.show()
```

### O que é o Boxplot e qual sua relação com a análise?

O **Boxplot (Diagrama de Caixa)** é um método gráfico para representar a distribuição de um conjunto de dados. Ele exibe um resumo estatístico dos dados, incluindo a mediana, os quartis e os outliers.

**Componentes do Boxplot:**

  - **A Caixa:** Representa o **Intervalo Interquartil (IQR)**, que contém os 50% centrais dos dados. A parte inferior da caixa é o primeiro quartil (Q1) e a parte superior é o terceiro quartil (Q3).
  - **A Linha na Caixa:** Marca a **mediana (Q2)**, que é o valor central dos dados.
  - **As Hastes (Whiskers):** As linhas que se estendem da caixa mostram a amplitude dos dados, tipicamente até 1.5 vezes o IQR.
  - **Os Pontos:** Pontos fora das hastes são considerados **outliers**.

**Como ele se relaciona com a análise?**
O boxplot é a ferramenta perfeita para avaliar a **estabilidade** do seu modelo sob as diferentes estratégias de validação. Enquanto o gráfico de barras mostra apenas a *média* da acurácia, o boxplot mostra a *distribuição completa*.

  - **Caixa mais curta e compacta:** Indica que os resultados da acurácia nos diferentes folds foram muito consistentes e variaram pouco. Isso sugere uma avaliação de modelo mais **estável e confiável**.
  - **Caixa mais longa:** Indica maior variabilidade nos resultados.

No projeto, é muito provável que o boxplot para o `RepeatedStratifiedKFold` seja mais compacto do que o do `KFold`, provando visualmente que ele é uma estratégia mais robusta para avaliar o desempenho do modelo.
