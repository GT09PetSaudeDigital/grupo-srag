"""
predizer_obito_paciente.py

Modelo de Machine Learning para prever óbito em casos de
SRAG, usando os dados do parquet do Guilherme.

  1) Pré-processamento respeitando a filosofia do projeto original
     (Sim / Não / Ignorado tratados como categorias distintas — coluna
     ausente NÃO é convertida em resultado negativo);
  2) PCA para reduzir a dimensionalidade da base antes de treinar;
  3) Comparação de 5 modelos: Regressão Logística, Random Forest,
     Gradient Boosting, HistGradientBoosting (mais forte, nativo do
     sklearn) e um StackingClassifier combinando os anteriores;
  4) Tratamento de desbalanceamento de classes (óbito é evento raro);
  5) Ajuste de limiar de decisão para maximizar F1 (em vez do 0.5 padrão);
  6) Avaliação com AUC-ROC, AUC-PR (mais informativa em classes raras),
     matriz de confusão e importância de variáveis via PCA loadings.

Uso:
    python predizer_obito_paciente.py --parquet-glob "data/parquet/srag/ano=*/srag.parquet"
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# Definindo o alvo da predição, que é o óbito
COL_EVOLUCAO = "EVOLUCAO"
VALORES_OBITO = {2, 3}
VALORES_CURA = {1}

# variáveis numéricas contínuas
COLUNAS_NUMERICAS = [
    "NU_IDADE_N",
]

# sintomas e comorbidades no padrão SIVEP (1=Sim, 2=Não, 9=Ignorado).
COLUNAS_SIM_NAO_IGNORADO = [
    "FEBRE", "TOSSE", "GARGANTA", "DISPNEIA", "DESC_RESP", "SATURACAO",
    "DIARREIA", "VOMITO",
    "CARDIOPATI", "PNEUMOPATI", "RENAL", "OBESIDADE", "DIABETES",
    "HEMATOLOGI", "HEPATICA", "ASMA", "IMUNODEPRE", "NEUROLOGIC",
    "PUERPERA", "SIND_DOWN",
    "UTI", "SUPORT_VEN",
]

# variáveis categóricas nominais (mais de 2 categorias relevantes)
COLUNAS_CATEGORICAS_NOMINAIS = [
    "CS_SEXO", "CS_RACA", "CS_ESCOL_N", "SG_UF_NOT",
]

VARIANCIA_ALVO_PCA = 0.90  # mantém componentes que expliquem 90% da variância
N_SPLITS_CV = 5
RANDOM_STATE = 42


# Carregamento dos dados

def carregar_dados(parquet_glob: str) -> pd.DataFrame:
    arquivos = sorted(glob.glob(parquet_glob))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo encontrado em '{parquet_glob}'. Rode "
            "scripts/ingest_all.py antes, ou ajuste --parquet-glob."
        )
    print(f"Lendo {len(arquivos)} arquivo(s) Parquet...")
    partes = [pd.read_parquet(a) for a in arquivos]
    df = pd.concat(partes, ignore_index=True)
    print(f"Total de registros carregados: {len(df):,}")
    return df


def montar_alvo(df: pd.DataFrame) -> pd.DataFrame:
    if COL_EVOLUCAO not in df.columns:
        raise KeyError(
            f"Coluna '{COL_EVOLUCAO}' não encontrada. Colunas disponíveis: "
            f"{list(df.columns)[:30]}..."
        )
    df = df.copy()
    # Processa os dados de evolucao e define OBITO = 1 quando evolucao for 2 e 3, e OBITO = 0 quando evolucao for 1, qualquer outro valor é descartado e removido, não sendo usado para treinamento ou teste.
    df["OBITO"] = np.select(
        [df[COL_EVOLUCAO].isin(VALORES_OBITO), df[COL_EVOLUCAO].isin(VALORES_CURA)],
        [1, 0],
        default=np.nan,
    )
    antes = len(df)
    df = df.dropna(subset=["OBITO"])
    df["OBITO"] = df["OBITO"].astype(int)
    print(
        f"Registros com evolução conhecida (cura/óbito): {len(df):,} "
        f"de {antes:,} ({len(df) / antes:.1%})"
    )
    print(f"Taxa de óbito na base: {df['OBITO'].mean():.2%}")
    return df


# 2) pré-processamento + PCA

def normalizar_tipos_categoricos(df: pd.DataFrame, colunas: list) -> pd.DataFrame:
    """
    Garante que cada coluna categórica tenha um único tipo (str), evitando o
    erro do OneHotEncoder quando a coluna mistura float/str/NaN (comum em
    Parquets vindos de CSV com códigos numéricos e valores ausentes).

    - Colunas numéricas (ex.: 1.0/2.0/9.0/NaN) viram "1"/"2"/"9"/NaN.
    - Colunas já-string mantêm o valor, com "nan"/"None" convertidos em NaN
      de verdade (para o SimpleImputer tratar corretamente depois).
    """
    df = df.copy()
    for col in colunas:
        if col not in df.columns:
            continue
        serie = df[col]

        # Tenta interpretar a coluna como códigos numéricos (ex.: 1/2/9),
        # mesmo quando o dtype é "object" com mistura de int/float/str
        # (comum em Parquet vindo de CSV com valores ausentes).
        coerced = pd.to_numeric(serie, errors="coerce")
        n_originais_nao_nulos = serie.notna().sum()
        n_convertidos = coerced.notna().sum()

        if n_originais_nao_nulos > 0 and n_convertidos / n_originais_nao_nulos > 0.95:
            # Coluna é essencialmente numérica -> unifica representação
            # (1, 1.0, "1", "1.0" viram todos "1"), preservando NaN.
            nova = coerced.astype("Int64").astype(str)
            nova = nova.replace("<NA>", np.nan)
        else:
            # Coluna genuinamente categórica em texto (ex.: "M"/"F", siglas de UF)
            nova = serie.astype(str)
            nova = nova.replace({"nan": np.nan, "None": np.nan, "NaT": np.nan, "<NA>": np.nan})

        df[col] = nova
    return df


def montar_preprocessador(df: pd.DataFrame):
    numericas = [c for c in COLUNAS_NUMERICAS if c in df.columns]
    sim_nao_ignorado = [c for c in COLUNAS_SIM_NAO_IGNORADO if c in df.columns]
    nominais = [c for c in COLUNAS_CATEGORICAS_NOMINAIS if c in df.columns]

    faltantes = (
        set(COLUNAS_NUMERICAS) - set(numericas)
        | set(COLUNAS_SIM_NAO_IGNORADO) - set(sim_nao_ignorado)
        | set(COLUNAS_CATEGORICAS_NOMINAIS) - set(nominais)
    )
    if faltantes:
        print(f"[aviso] colunas configuradas mas ausentes no Parquet, ignoradas: {faltantes}")

    transformador = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numericas,
            ),
            (
                # Sim/Não/Ignorado -> categórica (preserva "ignorado" como
                # categoria própria em vez de imputar como "não").
                "sni",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="9")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                sim_nao_ignorado,
            ),
            (
                "nom",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="DESCONHECIDO")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                nominais,
            ),
        ]
    )
    colunas_usadas = numericas + sim_nao_ignorado + nominais
    return transformador, colunas_usadas


# Modelos

def montar_modelos() -> dict:
    base_lr = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    )
    base_rf = RandomForestClassifier(
        n_estimators=400, max_depth=12, class_weight="balanced_subsample",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    base_gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
    )
    base_hgb = HistGradientBoostingClassifier(
        max_iter=400, max_depth=8, learning_rate=0.05,
        class_weight="balanced", random_state=RANDOM_STATE,
    )

    stacking = StackingClassifier(
        estimators=[
            ("random_forest", base_rf),
            ("gradient_boosting", base_gb),
            ("hist_gradient_boosting", base_hgb),
        ],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        stack_method="predict_proba",
        n_jobs=-1,
        cv=3,
    )

    return {
        "regressao_logistica": base_lr,
        "random_forest": base_rf,
        "gradient_boosting": base_gb,
        "hist_gradient_boosting": base_hgb,
        "stacking_ensemble": stacking,
    }


# Avaliação com validação cruzada

def avaliar_modelos(pre_processador, X, y) -> dict:
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",  # AUC-PR, melhor p/ classes raras
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
    }

    resultados = {}
    for nome, modelo in montar_modelos().items():
        pipeline = Pipeline([
            ("pre", pre_processador),
            ("pca", PCA(n_components=VARIANCIA_ALVO_PCA, svd_solver="full", random_state=RANDOM_STATE)),
            ("modelo", modelo),
        ])
        print(f"\nAvaliando: {nome} (validação cruzada {N_SPLITS_CV}-fold)...")
        scores = cross_validate(
            pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_estimator=False
        )
        resumo = {m: scores[f"test_{m}"].mean() for m in scoring}
        resultados[nome] = {"pipeline": pipeline, **resumo}
        print(
            f"  ROC-AUC={resumo['roc_auc']:.4f}  "
            f"AUC-PR={resumo['average_precision']:.4f}  "
            f"F1={resumo['f1']:.4f}  "
            f"Recall={resumo['recall']:.4f}  "
            f"Precision={resumo['precision']:.4f}"
        )

    return resultados


# Ajuste de limiar + avaliação final no conjunto de teste

def ajustar_limiar_e_avaliar(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    probas = pipeline.predict_proba(X_test)[:, 1]

    precisoes, recalls, limiares = precision_recall_curve(y_test, probas)
    f1s = 2 * (precisoes * recalls) / (precisoes + recalls + 1e-12)
    melhor_idx = np.nanargmax(f1s[:-1])  # último ponto não tem limiar correspondente
    melhor_limiar = limiares[melhor_idx]

    pred_padrao = (probas >= 0.5).astype(int)
    pred_ajustado = (probas >= melhor_limiar).astype(int)

    print(f"\nLimiar ótimo (maximiza F1): {melhor_limiar:.3f} (padrão seria 0.5)")
    print("\n--- Avaliação com limiar padrão (0.5) ---")
    print(classification_report(y_test, pred_padrao, target_names=["sobrevivente", "obito"]))

    print("--- Avaliação com limiar ajustado ---")
    print(classification_report(y_test, pred_ajustado, target_names=["sobrevivente", "obito"]))

    print(f"ROC-AUC (teste): {roc_auc_score(y_test, probas):.4f}")
    print(f"AUC-PR  (teste): {average_precision_score(y_test, probas):.4f}")

    matriz = confusion_matrix(y_test, pred_ajustado)
    print("\nMatriz de confusão (limiar ajustado):")
    print(pd.DataFrame(
        matriz,
        index=["real_sobrevivente", "real_obito"],
        columns=["previsto_sobrevivente", "previsto_obito"],
    ))

    return melhor_limiar, probas


# variância explicada pelo PCA e principais componentes

def relatorio_pca(pipeline):
    pca: PCA = pipeline.named_steps["pca"]
    n_componentes = pca.n_components_
    variancia_total = pca.explained_variance_ratio_.sum()
    print(
        f"\nPCA reduziu a base para {n_componentes} componentes, "
        f"explicando {variancia_total:.1%} da variância "
        f"(alvo configurado: {VARIANCIA_ALVO_PCA:.0%})."
    )

    try:
        nomes_features = pipeline.named_steps["pre"].get_feature_names_out()
        cargas = pd.DataFrame(
            pca.components_[:5],  # 5 primeiros componentes
            columns=nomes_features,
            index=[f"PC{i+1}" for i in range(min(5, n_componentes))],
        )
        print("\nVariáveis originais com maior peso nos 5 primeiros componentes:")
        for pc in cargas.index:
            top = cargas.loc[pc].abs().sort_values(ascending=False).head(5)
            print(f"  {pc}: {', '.join(top.index)}")
    except Exception as exc:  # pragma: no cover
        print(f"[aviso] não foi possível listar cargas do PCA: {exc}")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet-glob",
        default="data/parquet/srag/ano=*/srag.parquet",
        help="Padrão glob para localizar os arquivos Parquet processados.",
    )
    args = parser.parse_args()

    df = carregar_dados(args.parquet_glob)
    df = montar_alvo(df)

    pre_processador, colunas_usadas = montar_preprocessador(df)

    colunas_categoricas = [
        c for c in colunas_usadas
        if c in COLUNAS_SIM_NAO_IGNORADO or c in COLUNAS_CATEGORICAS_NOMINAIS
    ]
    df = normalizar_tipos_categoricos(df, colunas_categoricas)

    X = df[colunas_usadas]
    y = df["OBITO"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    resultados = avaliar_modelos(pre_processador, X_train, y_train)

    melhor_nome = max(resultados, key=lambda k: resultados[k]["roc_auc"])
    print(f"\n{'='*60}\nMelhor modelo por ROC-AUC (validação cruzada): {melhor_nome}\n{'='*60}")

    melhor_pipeline = resultados[melhor_nome]["pipeline"]
    limiar, _ = ajustar_limiar_e_avaliar(melhor_pipeline, X_train, y_train, X_test, y_test)
    relatorio_pca(melhor_pipeline)

    joblib.dump(
        {"pipeline": melhor_pipeline, "limiar_decisao": limiar, "colunas_usadas": colunas_usadas},
        "modelo_obito_srag_paciente.joblib",
    )
    print("\nModelo final salvo em modelo_obito_srag_paciente.joblib")
    print("(contém: pipeline completo pré-processamento+PCA+modelo, limiar de decisão e colunas usadas)")


if __name__ == "__main__":
    main()