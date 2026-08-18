"""Catálogo de features seguras para o modelo de admissão SRAG."""

DEMOGRAPHIC_FEATURES: tuple[str, ...] = (
    "CS_SEXO",
    "NU_IDADE_N",
    "CS_GESTANT",
)

SYMPTOM_FEATURES: tuple[str, ...] = (
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
)

COMORBIDITY_FEATURES: tuple[str, ...] = (
    "CARDIOPATI",
    "DIABETES",
    "PNEUMOPATI",
    "RENAL",
    "HEPATICA",
    "IMUNODEPRE",
    "OBESIDADE",
    "OUT_MORBI",
    "FATOR_RISC",
)

GEOGRAPHIC_FEATURES: tuple[str, ...] = (
    "SG_UF",
    "REGIAO",
)

TEMPORAL_FEATURES: tuple[str, ...] = (
    "SINT_ATE_NOTIF",
)

LEAKAGE_FEATURES: frozenset[str] = frozenset(
    {
        "EVOLUCAO",
        "DESFECHO_NORMALIZADO",
        "OBITO_SRAG",
        "DT_EVOLUCA",
        "UTI",
        "SUPORT_VEN",
        "QTD_DIAS",
        "DIAS_INTERNA",
        "PCR_EVOLUCAO",
    }
)

ADMISSION_FEATURES: tuple[str, ...] = (
    DEMOGRAPHIC_FEATURES
    + SYMPTOM_FEATURES
    + COMORBIDITY_FEATURES
    + GEOGRAPHIC_FEATURES
    + TEMPORAL_FEATURES
)


def validate_feature_registry() -> None:
    """Falha se uma feature proibida for incluída no modelo de admissão."""
    leaked = set(ADMISSION_FEATURES) & LEAKAGE_FEATURES
    if leaked:
        names = ", ".join(sorted(leaked))
        raise ValueError(f"Features com risco de leakage em ADMISSION_FEATURES: {names}")
