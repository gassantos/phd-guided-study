from pulp import (LpMinimize,
                  LpProblem,
                  LpVariable,
                  lpSum,
                  PULP_CBC_CMD)

# === Parâmetros do problema ===
# Número de tarefas (quantidade fixa a ser processada)
tasks = 100

# GPU A (rápida, mas cara e consome mais energia)
gpu_a_time = 1.5    # Tempo por tarefa (h)
gpu_a_cost = 2.5    # Custo por hora (USD)
gpu_a_energy = 300  # Consumo por hora (kWh)

# GPU B (mais lenta, barata e eficiente energeticamente)
gpu_b_time = 2.0    # Tempo por tarefa (h)
gpu_b_cost = 1.5    # Custo por hora (USD)
gpu_b_energy = 200  # Consumo por hora (kWh)

# Restrições globais
max_budget = 500    # Orçamento máximo (USD)
max_duration = 150  # Tempo máximo permitido (horas)

# Pesos para cada critério (ajuste os valores para priorizar critérios)
weight_time = 0.4
weight_cost = 0.3
weight_energy = 0.3

# === Variáveis de decisão ===
# Quantidade de GPUs A e B utilizadas
x1 = LpVariable("GPUs_A", lowBound=0, cat="Integer")
x2 = LpVariable("GPUs_B", lowBound=0, cat="Integer")

# === Modelo de otimização ===
# Criar o problema de otimização
model = LpProblem("Multiobjective_Optimization", LpMinimize)

# Função objetivo multiobjetivo (soma ponderada)
model += lpSum([
    weight_time * (x1 * gpu_a_time + x2 * gpu_b_time),
    weight_cost * (x1 * gpu_a_cost + x2 * gpu_b_cost),
    weight_energy * (x1 * gpu_a_energy + x2 * gpu_b_energy),
]), "Weighted_Objective"

# === Restrições ===
# Restrições de tempo máximo
model += (x1 * gpu_a_time + x2 * gpu_b_time <= max_duration), "Max_Duration"

# Restrições de orçamento
model += (x1 * gpu_a_cost + x2 * gpu_b_cost <= max_budget), "Max_Budget"

# Restrições para completar as tarefas
model += (x1 + x2 >= tasks), "Task_Completion"

# === Resolver o problema ===
# solver = "PULP_CBC_CMD"  (padrão PuLP)
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