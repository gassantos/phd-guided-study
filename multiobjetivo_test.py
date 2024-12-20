from pulp import LpMinimize
from pulp import lpSum
from pulp import LpVariable, LpProblem, PULP_CBC_CMD
from sklearn.preprocessing import MinMaxScaler


# === Dados do conjunto de instâncias ===
# Carregar dados (substituir pelo caminho dos dados processados previamente)
import pandas as pd
df_instances = pd.read_excel("data/doc_representacao_multiobjetivo.xlsx")

# === Normalização dos Dados ===
# Normalizar os dados para o intervalo [0, 1]
scaler = MinMaxScaler()
df_parametros = df_instances[["CUSTO_ENERGETICO", "CUSTO_FINANCEIRO", "TEMPO_EXECUCAO"]]
scaler = scaler.fit(df_parametros)
df_parametros = pd.DataFrame(scaler.transform(df_parametros), columns=df_parametros.columns)
print(df_parametros)


# Escolher os dados específicos para a instância
# Supondo que utilizaremos as médias dos custos e tempos para parametrização
mean_energy_cost = df_instances["CUSTO_ENERGETICO"].median()
mean_financial_cost = df_instances["CUSTO_FINANCEIRO"].median()
mean_execution_time = df_instances["TEMPO_EXECUCAO"].median()

# Parâmetros do problema baseados nas médias
tasks = 100
gpu_a_time = mean_execution_time * 0.75  # 75% da média para GPU A
gpu_a_cost = mean_financial_cost * 1.2   # 120% da média para GPU A
gpu_a_energy = mean_energy_cost * 1.3    # 130% da média para GPU A

gpu_b_time = mean_execution_time         # Média para GPU B
gpu_b_cost = mean_financial_cost         # Média para GPU B
gpu_b_energy = mean_energy_cost          # Média para GPU B

# Restrições globais (definidas para o problema)
max_budget = mean_financial_cost * 5
max_duration = mean_execution_time * 1.5

# Pesos para cada critério
weight_time = 0.4
weight_cost = 0.3
weight_energy = 0.3

# === Variáveis de decisão ===
x1 = LpVariable("GPUs_A", lowBound=0, cat="Integer")
x2 = LpVariable("GPUs_B", lowBound=0, cat="Integer")

# === Modelo de otimização ===
model = LpProblem("Multiobjective_Optimization", LpMinimize)

# Função objetivo multiobjetivo (soma ponderada)
model += lpSum([
    weight_time * (x1 * gpu_a_time + x2 * gpu_b_time),
    weight_cost * (x1 * gpu_a_cost + x2 * gpu_b_cost),
    weight_energy * (x1 * gpu_a_energy + x2 * gpu_b_energy),
]), "Weighted_Objective"

# === Restrições ===
model += (x1 * gpu_a_time + x2 * gpu_b_time <= max_duration), "Max_Duration"
model += (x1 * gpu_a_cost + x2 * gpu_b_cost <= max_budget), "Max_Budget"
model += (x1 + x2 >= tasks), "Task_Completion"

# === Resolver o problema ===
model.solve(PULP_CBC_CMD())

# === Exibir os resultados ===
print("Status:", model.status)
print("Solução Ótima:")
print(f"  GPUs A (x1): {x1.varValue}")
print(f"  GPUs B (x2): {x2.varValue}")
print("\nFunções Objetivo Calculadas:")
print(f"  Tempo Total: {x1.varValue * gpu_a_time + x2.varValue * gpu_b_time:.2f} horas")
print(f"  Custo Total: {x1.varValue * gpu_a_cost + x2.varValue * gpu_b_cost:.2f} USD")
print(f"  Consumo Energético Total: {x1.varValue * gpu_a_energy + x2.varValue * gpu_b_energy:.2f} kWh")
