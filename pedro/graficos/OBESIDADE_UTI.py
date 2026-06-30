import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./INFLUD20-23-03-2026.csv', sep=';', low_memory=False)

df_uti = df[df['OBESIDADE'].isin([1,2]) & df['UTI'].isin([1,2])].copy()
df_uti['OBESIDADE_LABEL'] = df_uti['OBESIDADE'].map({1:'Obeso', 2:'Não obeso'})
df_uti['UTI_LABEL'] = df_uti['UTI'].map({1:'Sim', 2:'Não'})

tab_uti = pd.crosstab(df_uti['OBESIDADE_LABEL'], df_uti['UTI_LABEL'], normalize='index') * 100
print("\nObesidade x UTI (%):")
print(tab_uti.round(2))

tab_uti.plot(kind='bar', stacked=True, figsize=(8,5))
plt.title('Obesidade x UTI')
plt.ylabel('Percentual (%)')
plt.xlabel('')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()