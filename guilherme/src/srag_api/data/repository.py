from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import duckdb


@dataclass(frozen=True, slots=True)
class SragFilters:
    ano_inicio: int | None = None
    ano_fim: int | None = None
    uf: str | None = None
    municipio: str | None = None
    codigo_municipio: int | None = None
    sexo: str | None = None
    faixa_etaria: str | None = None
    etiologia: str | None = None


class SragRepository:
    """Consultas somente-leitura sobre o dataset Parquet SRAG."""

    VIEW_NAME = "srag"

    def __init__(self, parquet_root: Path):
        self.parquet_root = Path(parquet_root)

    def _parquet_files(self) -> list[str]:
        files = sorted(
            self.parquet_root.glob("srag/ano=*/srag.parquet")
        )
        if not files:
            raise FileNotFoundError(
                f"Nenhum Parquet SRAG encontrado em {self.parquet_root}"
            )
        return [str(path.resolve()) for path in files]

    def _connect(self) -> duckdb.DuckDBPyConnection:
        connection = duckdb.connect(database=":memory:")
        relation = connection.read_parquet(
            self._parquet_files(),
            union_by_name=True,
            hive_partitioning=False,
        )
        relation.create_view(self.VIEW_NAME)
        return connection

    @staticmethod
    def _where(filters: SragFilters) -> tuple[str, list[object]]:
        clauses: list[str] = []
        params: list[object] = []

        if filters.ano_inicio is not None:
            clauses.append("ANO >= ?")
            params.append(filters.ano_inicio)

        if filters.ano_fim is not None:
            clauses.append("ANO <= ?")
            params.append(filters.ano_fim)

        if filters.uf:
            clauses.append("UF = ?")
            params.append(filters.uf.strip().upper())

        if filters.municipio:
            if not filters.uf and filters.codigo_municipio is None:
                raise ValueError(
                    "Filtro por municipio exige UF ou codigo_municipio."
                )
            clauses.append("MUNICIPIO = ?")
            params.append(filters.municipio.strip().upper())

        if filters.codigo_municipio is not None:
            clauses.append("CODIGO_MUNICIPIO = ?")
            params.append(filters.codigo_municipio)

        if filters.sexo:
            clauses.append("CS_SEXO = ?")
            params.append(filters.sexo.strip().upper())

        if filters.faixa_etaria:
            clauses.append("FAIXA_ETARIA = ?")
            params.append(filters.faixa_etaria)

        if filters.etiologia:
            clauses.append("ETIOLOGIA_NORMALIZADA = ?")
            params.append(filters.etiologia)

        if not clauses:
            return "", params

        return " WHERE " + " AND ".join(clauses), params

    @staticmethod
    def _rows_to_dicts(
        connection: duckdb.DuckDBPyConnection,
    ) -> list[dict[str, object]]:
        columns = [item[0] for item in connection.description]
        return [
            dict(zip(columns, row))
            for row in connection.fetchall()
        ]

    def get_total_cases(
        self,
        filters: SragFilters = SragFilters(),
    ) -> int:
        where, params = self._where(filters)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM {self.VIEW_NAME}{where}",
                params,
            ).fetchone()
        return int(row[0])

    def get_deaths(
        self,
        filters: SragFilters = SragFilters(),
    ) -> int:
        where, params = self._where(filters)
        extra = "DESFECHO_NORMALIZADO = 'OBITO_SRAG'"
        separator = " AND " if where else " WHERE "
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM {self.VIEW_NAME}"
                f"{where}{separator}{extra}",
                params,
            ).fetchone()
        return int(row[0])

    def get_known_outcome_count(
        self,
        filters: SragFilters = SragFilters(),
    ) -> int:
        where, params = self._where(filters)
        extra = (
            "DESFECHO_NORMALIZADO IN "
            "('CURA', 'OBITO_SRAG', 'OBITO_OUTRAS_CAUSAS')"
        )
        separator = " AND " if where else " WHERE "
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM {self.VIEW_NAME}"
                f"{where}{separator}{extra}",
                params,
            ).fetchone()
        return int(row[0])

    def get_unknown_outcome_count(
        self,
        filters: SragFilters = SragFilters(),
    ) -> int:
        where, params = self._where(filters)
        extra = "DESFECHO_NORMALIZADO IN ('IGNORADO', 'AUSENTE', 'OUTRO')"
        separator = " AND " if where else " WHERE "
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM {self.VIEW_NAME}"
                f"{where}{separator}{extra}",
                params,
            ).fetchone()
        return int(row[0])

    def get_icu_cases(
        self,
        filters: SragFilters = SragFilters(),
    ) -> int:
        where, params = self._where(filters)
        extra = "FOI_UTI = 'SIM'"
        separator = " AND " if where else " WHERE "
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM {self.VIEW_NAME}"
                f"{where}{separator}{extra}",
                params,
            ).fetchone()
        return int(row[0])

    def get_known_icu_count(
        self,
        filters: SragFilters = SragFilters(),
    ) -> int:
        where, params = self._where(filters)
        extra = "FOI_UTI IN ('SIM', 'NAO')"
        separator = " AND " if where else " WHERE "
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM {self.VIEW_NAME}"
                f"{where}{separator}{extra}",
                params,
            ).fetchone()
        return int(row[0])

    def get_unknown_icu_count(
        self,
        filters: SragFilters = SragFilters(),
    ) -> int:
        where, params = self._where(filters)
        extra = "FOI_UTI IN ('IGNORADO', 'AUSENTE')"
        separator = " AND " if where else " WHERE "
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM {self.VIEW_NAME}"
                f"{where}{separator}{extra}",
                params,
            ).fetchone()
        return int(row[0])

    def get_age_distribution(
        self,
        filters: SragFilters = SragFilters(),
    ) -> list[dict[str, object]]:
        where, params = self._where(filters)
        sql = f"""
            SELECT FAIXA_ETARIA AS faixa_etaria, COUNT(*) AS casos
            FROM {self.VIEW_NAME}
            {where}
            GROUP BY FAIXA_ETARIA
            ORDER BY MIN(IDADE_ANOS) NULLS LAST
        """
        with self._connect() as connection:
            result = connection.execute(sql, params)
            return self._rows_to_dicts(result)

    def get_etiology_distribution(
        self,
        filters: SragFilters = SragFilters(),
    ) -> list[dict[str, object]]:
        where, params = self._where(filters)
        sql = f"""
            SELECT ETIOLOGIA_NORMALIZADA AS etiologia, COUNT(*) AS casos
            FROM {self.VIEW_NAME}
            {where}
            GROUP BY ETIOLOGIA_NORMALIZADA
            ORDER BY casos DESC, etiologia
        """
        with self._connect() as connection:
            result = connection.execute(sql, params)
            return self._rows_to_dicts(result)

    def get_available_columns(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                f"DESCRIBE {self.VIEW_NAME}"
            ).fetchall()

        return {str(row[0]).upper() for row in rows}

    def get_comorbidity_distribution(
        self,
        filters: SragFilters = SragFilters(),
        column_map: dict[str, str] | None = None,
    ) -> list[dict[str, object]]:
        if not column_map:
            return []

        where, params = self._where(filters)
        separator = " AND " if where else " WHERE "
        results: list[dict[str, object]] = []

        with self._connect() as connection:
            rows = connection.execute(
                f"DESCRIBE {self.VIEW_NAME}"
            ).fetchall()
            available = {str(row[0]).upper() for row in rows}

            for column, label in column_map.items():
                safe_column = column.upper()
                if safe_column not in available:
                    continue

                quoted_column = '"' + safe_column.replace('"', '""') + '"'
                row = connection.execute(
                    f"SELECT COUNT(*) FROM {self.VIEW_NAME}"
                    f"{where}{separator}{quoted_column} = 1",
                    params,
                ).fetchone()
                count = int(row[0])
                if count > 0:
                    results.append({
                        "comorbidade": label,
                        "casos": count,
                    })

        return sorted(
            results,
            key=lambda item: (-int(item["casos"]), str(item["comorbidade"])),
        )

    def get_time_series(
        self,
        filters: SragFilters = SragFilters(),
        frequency: Literal["month", "week"] = "month",
    ) -> list[dict[str, object]]:
        where, params = self._where(filters)

        if frequency == "month":
            period_column = "MES"
            period_alias = "mes"
        elif frequency == "week":
            period_column = "SEMANA_EPIDEMIOLOGICA"
            period_alias = "semana_epidemiologica"
        else:
            raise ValueError("frequency deve ser 'month' ou 'week'.")

        null_clause = f"{period_column} IS NOT NULL"
        separator = " AND " if where else " WHERE "

        sql = f"""
            SELECT
                ANO AS ano,
                {period_column} AS {period_alias},
                COUNT(*) AS casos
            FROM {self.VIEW_NAME}
            {where}{separator}{null_clause}
            GROUP BY ANO, {period_column}
            ORDER BY ANO, {period_column}
        """
        with self._connect() as connection:
            result = connection.execute(sql, params)
            return self._rows_to_dicts(result)

    def get_ranking(
        self,
        filters: SragFilters = SragFilters(),
        level: Literal["uf", "municipio"] = "uf",
        metric: Literal["cases", "deaths", "icu"] = "cases",
        limit: int = 20,
    ) -> list[dict[str, object]]:
        if not 1 <= limit <= 100:
            raise ValueError("limit deve estar entre 1 e 100.")

        if level == "uf":
            group_columns = "UF"
            select_columns = "UF AS uf"
        elif level == "municipio":
            group_columns = "UF, CODIGO_MUNICIPIO, MUNICIPIO"
            select_columns = (
                "UF AS uf, "
                "CODIGO_MUNICIPIO AS codigo_municipio, "
                "MUNICIPIO AS municipio"
            )
        else:
            raise ValueError("level deve ser 'uf' ou 'municipio'.")

        if metric == "cases":
            metric_expression = "COUNT(*)"
        elif metric == "deaths":
            metric_expression = (
                "SUM(CASE WHEN DESFECHO_NORMALIZADO = 'OBITO_SRAG' "
                "THEN 1 ELSE 0 END)"
            )
        elif metric == "icu":
            metric_expression = (
                "SUM(CASE WHEN FOI_UTI = 'SIM' THEN 1 ELSE 0 END)"
            )
        else:
            raise ValueError("metric deve ser 'cases', 'deaths' ou 'icu'.")

        where, params = self._where(filters)
        params = [*params, limit]

        sql = f"""
            SELECT
                {select_columns},
                {metric_expression} AS valor
            FROM {self.VIEW_NAME}
            {where}
            GROUP BY {group_columns}
            ORDER BY valor DESC
            LIMIT ?
        """

        with self._connect() as connection:
            result = connection.execute(sql, params)
            return self._rows_to_dicts(result)
