"""
=====================================================================
 ANÁLISE SRAG (Síndrome Respiratória Aguda Grave) - Foco Paraná (PR)
=====================================================================
Fonte dos dados: https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026
Sistema: SIVEP-Gripe / Ministério da Saúde

O QUE ESSE SCRIPT FAZ:
  1. Baixa (ou lê localmente) o CSV da base SRAG de um ano escolhido
  2. Limpa e trata os dados (valores ausentes, tipos, datas)
  3. Filtra os registros do Paraná (UF de residência = PR)
  4. Gera gráficos (EDA) sobre a base geral e sobre o Paraná
  5. Roda algoritmos de Machine Learning para prever:
       - Óbito (variável EVOLUCAO)
       - Internação em UTI (variável UTI)
     usando Regressão Logística, Random Forest e Gradient Boosting,
     comparando as métricas (acurácia, precisão, recall, F1, AUC).

COMO USAR:
  1. (Opcional) Baixe o CSV manualmente em:
     https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026
     -> escolha "Banco vivo - CSV" do ano desejado (ex: 2025)
     e salve como, por exemplo, "INFLUD25.csv" na mesma pasta do script.

  2. Rode:
     python analise_srag_pr.py --arquivo INFLUD25.csv --uf PR

  3. Se NÃO passar --arquivo, o script tenta baixar automaticamente
     (precisa de internet liberada para dadosabertos.saude.gov.br).

  Os gráficos (.png) e o relatório (.txt) são salvos na pasta "saida/".

REQUISITOS:
  pip install pandas numpy matplotlib seaborn scikit-learn requests
=====================================================================
"""

import argparse
import os
import sys
import warnings
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # gera imagens sem precisar de tela
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")


# Fonte: https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026

URLS_POR_ANO = {
    2019: "https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026/resource/d96d6348-083a-4184-a39a-794b5e8ec337",
    2020: "https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026/resource/4b877048-d59c-4d29-b4e8-f715e05fda80",
    2021: "https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026/resource/47743cf4-2b93-4161-905d-838bce5f4961",
    2022: "https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026/resource/40d4027f-825e-4acd-98d8-21e0af36a183",
    2023: "https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026/resource/0d78ff63-d6ca-4311-8dc8-6123cf1ca127",
    2024: "https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026/resource/8cb52f73-0184-41d5-8a8f-87d8f415652c",
    2025: "https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026/resource/20c49de3-ddc3-4b76-a942-1518eaae9c91",
    2026: "https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026/resource/74091efc-3f75-42e8-a6fa-6b79a8d30582",
}

PASTA_SAIDA = "saida"

def baixar_manual_info(ano):
    print(f"\n[AVISO] Não consegui baixar automaticamente o CSV de {ano}.")
    print("Baixe manualmente em:")
    print(f"  {URLS_POR_ANO.get(ano, 'https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026')}")
    print("Depois rode novamente com: --arquivo CAMINHO_DO_CSV.csv\n")
    sys.exit(1)


def carregar_dados(caminho_arquivo, ano):
    if caminho_arquivo and os.path.exists(caminho_arquivo):
        print(f"Lendo arquivo local: {caminho_arquivo}")
        # A base é grande e usa ; como separador (padrão DATASUS)
        df = pd.read_csv(caminho_arquivo, sep=";", encoding="latin-1",
                          low_memory=False)
        return df

    print(f"Tentando baixar automaticamente a base de {ano}...")
    try:
        url_pagina = URLS_POR_ANO.get(ano)
        if url_pagina is None:
            raise ValueError("Ano não mapeado")
        raise RuntimeError("Download automático não configurado - use --arquivo")
    except Exception as e:
        print(f"Falha: {e}")
        baixar_manual_info(ano)


# ---------------------------------------------------------------
# 1) LIMPEZA E PRÉ-PROCESSAMENTO
# ---------------------------------------------------------------
def limpar_dados(df):
    print("\n=== LIMPEZA DE DADOS ===")
    print(f"Registros originais: {len(df):,}")

    df.columns = [c.strip().upper() for c in df.columns]

    # Datas relevantes
    colunas_data = [c for c in ["DT_NOTIFIC", "DT_SIN_PRI", "DT_EVOLUCA",
                                 "DT_INTERNA", "DT_ENTUTI"] if c in df.columns]
    for c in colunas_data:
        df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    # Remove duplicados
    antes = len(df)
    df = df.drop_duplicates()
    print(f"Duplicados removidos: {antes - len(df):,}")

    # Idade: normaliza para anos (NU_IDADE_N no SIVEP já costuma vir em anos
    # quando TP_IDADE == 3; valores muito fora do razoável são tratados)
    if "NU_IDADE_N" in df.columns:
        df["NU_IDADE_N"] = pd.to_numeric(df["NU_IDADE_N"], errors="coerce")
        df.loc[(df["NU_IDADE_N"] < 0) | (df["NU_IDADE_N"] > 120), "NU_IDADE_N"] = np.nan

    # Sexo: padroniza
    if "CS_SEXO" in df.columns:
        df["CS_SEXO"] = df["CS_SEXO"].replace({"I": np.nan})

    # Evolução do caso (1=Cura, 2=Óbito, 3=Óbito por outras causas, 9=Ignorado)
    if "EVOLUCAO" in df.columns:
        df["EVOLUCAO"] = pd.to_numeric(df["EVOLUCAO"], errors="coerce")

    if "UTI" in df.columns:
        df["UTI"] = pd.to_numeric(df["UTI"], errors="coerce")

    print(f"Registros após limpeza básica: {len(df):,}")
    return df


# ---------------------------------------------------------------
# 2) FILTRO PARANÁ
# ---------------------------------------------------------------
def filtrar_uf(df, uf="PR"):
    col_uf = None
    for candidato in ["SG_UF", "SG_UF_NOT"]:
        if candidato in df.columns:
            col_uf = candidato
            break
    if col_uf is None:
        print("[AVISO] Coluna de UF não encontrada - pulando filtro geográfico.")
        return df

    df_uf = df[df[col_uf].astype(str).str.upper() == uf].copy()
    print(f"\nRegistros filtrados para {uf} (coluna {col_uf}): {len(df_uf):,} "
          f"de {len(df):,} ({len(df_uf)/len(df)*100:.1f}%)")
    return df_uf


# ---------------------------------------------------------------
# 3) VISUALIZAÇÕES (EDA)
# ---------------------------------------------------------------
def gerar_graficos(df_brasil, df_pr, uf="PR"):
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    print("\n=== GERANDO GRÁFICOS ===")

    # --- 1. Casos ao longo do tempo (Brasil x PR) ---
    if "DT_SIN_PRI" in df_brasil.columns:
        fig, ax = plt.subplots(figsize=(11, 5))
        serie_br = df_brasil.set_index("DT_SIN_PRI").resample("ME").size()
        serie_pr = df_pr.set_index("DT_SIN_PRI").resample("ME").size()
        ax.plot(serie_br.index, serie_br.values, label="Brasil", linewidth=2)
        ax.plot(serie_pr.index, serie_pr.values, label=uf, linewidth=2)
        ax.set_title("Casos de SRAG por mês (início dos sintomas)")
        ax.set_xlabel("Mês"); ax.set_ylabel("Número de casos")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{PASTA_SAIDA}/01_casos_ao_longo_do_tempo.png", dpi=150)
        plt.close()

    # --- 2. Distribuição de idade (PR) ---
    if "NU_IDADE_N" in df_pr.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(df_pr["NU_IDADE_N"].dropna(), bins=30, kde=True, ax=ax, color="#4C72B0")
        ax.set_title(f"Distribuição de idade dos casos de SRAG - {uf}")
        ax.set_xlabel("Idade (anos)"); ax.set_ylabel("Frequência")
        plt.tight_layout()
        plt.savefig(f"{PASTA_SAIDA}/02_distribuicao_idade_pr.png", dpi=150)
        plt.close()

    # --- 3. Sexo (PR) ---
    if "CS_SEXO" in df_pr.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        df_pr["CS_SEXO"].value_counts().plot(kind="bar", ax=ax, color=["#55A868", "#C44E52"])
        ax.set_title(f"Casos de SRAG por sexo - {uf}")
        ax.set_xlabel("Sexo"); ax.set_ylabel("Número de casos")
        plt.tight_layout()
        plt.savefig(f"{PASTA_SAIDA}/03_casos_por_sexo_pr.png", dpi=150)
        plt.close()

    # --- 4. Evolução do caso (Cura x Óbito) PR ---
    if "EVOLUCAO" in df_pr.columns:
        mapa_evol = {1: "Cura", 2: "Óbito", 3: "Óbito (outras causas)", 9: "Ignorado"}
        contagem = df_pr["EVOLUCAO"].map(mapa_evol).value_counts()
        fig, ax = plt.subplots(figsize=(7, 5))
        contagem.plot(kind="bar", ax=ax, color="#8172B2")
        ax.set_title(f"Evolução dos casos de SRAG - {uf}")
        ax.set_xlabel("Evolução"); ax.set_ylabel("Número de casos")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(f"{PASTA_SAIDA}/04_evolucao_casos_pr.png", dpi=150)
        plt.close()

    # --- 5. Taxa de UTI por faixa etária (PR) ---
    if "UTI" in df_pr.columns and "NU_IDADE_N" in df_pr.columns:
        tmp = df_pr.dropna(subset=["UTI", "NU_IDADE_N"]).copy()
        tmp["FAIXA_ETARIA"] = pd.cut(
            tmp["NU_IDADE_N"], bins=[0, 12, 18, 30, 45, 60, 75, 120],
            labels=["0-12", "13-18", "19-30", "31-45", "46-60", "61-75", "76+"]
        )
        taxa_uti = tmp.groupby("FAIXA_ETARIA", observed=True)["UTI"].apply(
            lambda s: (s == 1).mean() * 100
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        taxa_uti.plot(kind="bar", ax=ax, color="#DD8452")
        ax.set_title(f"Taxa de internação em UTI por faixa etária - {uf}")
        ax.set_xlabel("Faixa etária"); ax.set_ylabel("% internados em UTI")
        plt.tight_layout()
        plt.savefig(f"{PASTA_SAIDA}/05_taxa_uti_faixa_etaria_pr.png", dpi=150)
        plt.close()

    # --- 6. Comorbidades mais comuns (PR) ---
    comorbidades = ["CARDIOPATI", "DIABETES", "OBESIDADE", "PNEUMOPATI",
                     "RENAL", "OBESIDADE", "HEPATICA", "NEUROLOGIC", "ASMA"]
    presentes = [c for c in comorbidades if c in df_pr.columns]
    if presentes:
        proporcoes = {}
        for c in presentes:
            serie = pd.to_numeric(df_pr[c], errors="coerce")
            proporcoes[c] = (serie == 1).mean() * 100
        fig, ax = plt.subplots(figsize=(9, 5))
        pd.Series(proporcoes).sort_values(ascending=True).plot(kind="barh", ax=ax, color="#64B5CD")
        ax.set_title(f"Prevalência de comorbidades entre casos de SRAG - {uf}")
        ax.set_xlabel("% dos casos com a comorbidade")
        plt.tight_layout()
        plt.savefig(f"{PASTA_SAIDA}/06_comorbidades_pr.png", dpi=150)
        plt.close()

    # --- 7. Top municípios do PR com mais casos ---
    col_mun = "ID_MUNICIP" if "ID_MUNICIP" in df_pr.columns else None
    if col_mun:
        fig, ax = plt.subplots(figsize=(9, 6))
        df_pr[col_mun].value_counts().head(15).sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_title(f"Top 15 municípios do {uf} com mais notificações de SRAG")
        ax.set_xlabel("Número de casos")
        plt.tight_layout()
        plt.savefig(f"{PASTA_SAIDA}/07_top_municipios_pr.png", dpi=150)
        plt.close()

    print(f"Gráficos salvos em ./{PASTA_SAIDA}/")


# ---------------------------------------------------------------
# 4) MACHINE LEARNING
# ---------------------------------------------------------------
def preparar_features_ml(df, alvo="EVOLUCAO"):
    """Monta a matriz de features numéricas/categóricas simples para o ML."""
    candidatas = [
        "NU_IDADE_N", "CS_SEXO", "CS_RACA", "CS_ESCOL_N", "VACINA_COV",
        "FEBRE", "TOSSE", "GARGANTA", "DISPNEIA", "DESC_RESP", "SATURACAO",
        "DIARREIA", "VOMITO", "CARDIOPATI", "DIABETES", "OBESIDADE",
        "PNEUMOPATI", "RENAL", "HEPATICA", "NEUROLOGIC", "ASMA", "PUERPERA",
        "SG_UF_NOT",
    ]
    cols = [c for c in candidatas if c in df.columns]
    if alvo not in df.columns:
        return None, None, None

    base = df[cols + [alvo]].copy()

    # alvo binário
    if alvo == "EVOLUCAO":
        base = base[base[alvo].isin([1, 2])]  # cura x óbito (remove ignorado/outras causas)
        base["ALVO"] = (base[alvo] == 2).astype(int)  # 1 = óbito
    elif alvo == "UTI":
        base = base[base[alvo].isin([1, 2])]
        base["ALVO"] = (base[alvo] == 1).astype(int)  # 1 = foi p/ UTI
    base = base.drop(columns=[alvo])

    y = base["ALVO"]
    X = base.drop(columns=["ALVO"])

    # separa numéricas e categóricas
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # imputação simples
    if num_cols:
        X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
    if cat_cols:
        X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y, num_cols


def rodar_ml(df_pr, alvo="EVOLUCAO", nome_alvo="Óbito"):
    print(f"\n=== MACHINE LEARNING: prevendo {nome_alvo} ({alvo}) - Paraná ===")
    X, y, num_cols = preparar_features_ml(df_pr, alvo=alvo)
    if X is None or len(X) < 100 or y.nunique() < 2:
        print("Dados insuficientes para treinar modelo para este alvo. Pulando.")
        return None

    print(f"Amostras: {len(X):,} | Positivos ({nome_alvo}): {y.sum():,} ({y.mean()*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    modelos = {
        "Regressão Logística": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12,
                                                 class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
    }

    resultados = []
    plt.figure(figsize=(7, 6))
    for nome, modelo in modelos.items():
        if nome == "Regressão Logística":
            modelo.fit(X_train_s, y_train)
            probs = modelo.predict_proba(X_test_s)[:, 1]
            preds = modelo.predict(X_test_s)
        else:
            modelo.fit(X_train, y_train)
            probs = modelo.predict_proba(X_test)[:, 1]
            preds = modelo.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)

        resultados.append({"Modelo": nome, "Acurácia": acc, "Precisão": prec,
                            "Recall": rec, "F1": f1, "AUC": auc})

        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{nome} (AUC={auc:.2f})")

        # matriz de confusão individual
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=["Não", "Sim"], yticklabels=["Não", "Sim"])
        ax_cm.set_title(f"Matriz de Confusão - {nome}\n(alvo: {nome_alvo})")
        ax_cm.set_xlabel("Previsto"); ax_cm.set_ylabel("Real")
        plt.tight_layout()
        nome_arq = nome.lower().replace(" ", "_").replace("í", "i").replace("ã", "a")
        plt.savefig(f"{PASTA_SAIDA}/ml_matriz_confusao_{alvo}_{nome_arq}.png", dpi=150)
        plt.close(fig_cm)

        # importância de features para os modelos de árvore
        if hasattr(modelo, "feature_importances_"):
            imp = pd.Series(modelo.feature_importances_, index=X.columns)
            imp = imp.sort_values(ascending=False).head(15)
            fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
            imp.sort_values().plot(kind="barh", ax=ax_imp, color="#55A868")
            ax_imp.set_title(f"Importância das variáveis - {nome} ({nome_alvo})")
            plt.tight_layout()
            plt.savefig(f"{PASTA_SAIDA}/ml_importancia_{alvo}_{nome_arq}.png", dpi=150)
            plt.close(fig_imp)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"Curvas ROC - previsão de {nome_alvo} (Paraná)")
    plt.xlabel("Falso Positivo"); plt.ylabel("Verdadeiro Positivo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PASTA_SAIDA}/ml_curva_roc_{alvo}.png", dpi=150)
    plt.close()

    df_result = pd.DataFrame(resultados)
    print(df_result.to_string(index=False))
    df_result.to_csv(f"{PASTA_SAIDA}/ml_resultados_{alvo}.csv", index=False)
    return df_result


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Análise SRAG - foco Paraná")
    parser.add_argument("--arquivo", type=str, default=None,
                         help="Caminho do CSV baixado do portal SRAG")
    parser.add_argument("--ano", type=int, default=2025)
    parser.add_argument("--uf", type=str, default="PR")
    args = parser.parse_args()

    df = carregar_dados(args.arquivo, args.ano)
    df = limpar_dados(df)
    df_uf = filtrar_uf(df, uf=args.uf)

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    gerar_graficos(df, df_uf, uf=args.uf)

    resultado_obito = rodar_ml(df_uf, alvo="EVOLUCAO", nome_alvo="Óbito")
    resultado_uti = rodar_ml(df_uf, alvo="UTI", nome_alvo="Internação em UTI")

    print("\n=== CONCLUÍDO ===")
    print(f"Todos os gráficos e resultados estão na pasta ./{PASTA_SAIDA}/")


if __name__ == "__main__":
    main()