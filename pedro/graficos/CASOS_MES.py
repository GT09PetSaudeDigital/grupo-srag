import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./INFLUD20-23-03-2026.csv', sep=';', low_memory=False)

df['DT_SIN_PRI'] = pd.to_datetime(df['DT_SIN_PRI'], errors='coerce')

df = df.dropna(subset=['DT_SIN_PRI'])

df['ANO_MES_CASOS'] = df['DT_SIN_PRI'].dt.to_period('M')

casos_por_mes = df.groupby('ANO_MES').size()
obitos_por_mes = df.groupby('ANO_MES').size()

casos_por_mes.index = casos_por_mes.index.astype(str)

plt.figure(figsize=(12,6))
plt.plot(casos_por_mes.index, casos_por_mes.values, marker='o')
plt.title('Quantidade de casos por mês')
plt.xlabel('Mês')
plt.ylabel('Número de casos')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()