import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    '/Users/gabrielribas/pedroribas/PET-Saude/INFLUD20-23-03-2026.csv',
    sep=';',
    low_memory=False
)

# converter DT_SIN_PRI para data
df['DT_SIN_PRI'] = pd.to_datetime(df['DT_SIN_PRI'], errors='coerce')

# remover registros sem data válida
df = df.dropna(subset=['DT_SIN_PRI'])

# criar coluna ano-mês com base em DT_SIN_PRI
df['ANO_MES'] = df['DT_SIN_PRI'].dt.to_period('M')

# casos por mês
casos_por_mes = df.groupby('ANO_MES').size()

# óbitos por mês
obitos_por_mes = df[df['EVOLUCAO'] == 2].groupby('ANO_MES').size()

# juntar as duas séries em uma tabela
serie_mensal = pd.DataFrame({
    'Casos': casos_por_mes,
    'Óbitos': obitos_por_mes
}).fillna(0)

# transformar em inteiro
serie_mensal = serie_mensal.astype(int)

# ordenar por mês
serie_mensal = serie_mensal.sort_index()

# converter índice para string para exibir no gráfico
serie_mensal.index = serie_mensal.index.astype(str)

plt.figure(figsize=(14,6))
plt.plot(serie_mensal.index, serie_mensal['Casos'], marker='o', label='Casos')
plt.plot(serie_mensal.index, serie_mensal['Óbitos'], marker='o', label='Óbitos')

plt.title('Casos e óbitos por mês (base: DT_SIN_PRI)')
plt.xlabel('Mês')
plt.ylabel('Quantidade')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()