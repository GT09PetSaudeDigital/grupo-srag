import pandas as pd

df = pd.read_csv(
    "/Users/gabrielribas/pedroribas/PET-Saude/INFLUD20-23-03-2026.csv",
    sep=";",
    low_memory=False
)

colunas = [
    # datas
    "DT_SIN_PRI",
    "DT_INTERNA",
    "DT_ENTUTI",
    "DT_SAIDUTI",
    "DT_EVOLUCA",

    # Dados demográficos
    "CS_SEXO",
    "NU_IDADE_N",
    "TP_IDADE",
    "CS_RACA",
    "CS_ESCOL_N",
    "CS_GESTANT",
    "PUERPERA",

    # Localização
    "SG_UF_NOT",

    # Sintomas
    "FEBRE",
    "TOSSE",
    "GARGANTA",
    "DISPNEIA",
    "DESC_RESP",
    "SATURACAO",
    "DIARREIA",
    "VOMITO",
    "DOR_ABD",
    "FADIGA",
    "PERD_OLFT",
    "PERD_PALA",
    "OUTRO_SIN",

    # Comorbidades
    "CARDIOPATI",
    "HEMATOLOGI",
    "SIND_DOWN",
    "HEPATICA",
    "ASMA",
    "DIABETES",
    "NEUROLOGIC",
    "PNEUMOPATI",
    "IMUNODEPRE",
    "RENAL",
    "OBESIDADE",
    "TABAG",
    "OUT_MORBI",

    # Gravidade
    "HOSPITAL",
    "UTI",
    "SUPORT_VEN",
    "EVOLUCAO",

    # Diagnóstico
    "CLASSI_FIN",
    "CRITERIO",
    "PCR_RESUL",
    "PCR_SARS2"
]

colunas = [c for c in colunas if c in df.columns]

df = df[colunas]

df.to_csv("INFLUD20_reduzido.csv", sep=";", index=False)