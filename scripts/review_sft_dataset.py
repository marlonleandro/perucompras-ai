#!/usr/bin/env python3
"""PeruCompras AI / E02 v2.1: round-trip de revisión humana con Excel.

Uso desde la raíz del proyecto:
  python scripts/review_sft_dataset.py export \
      --input data/sft/e02_v2_1/review_candidates.jsonl \
      --excel data/sft/e02_v2_1/review_e02.xlsx
  python scripts/review_sft_dataset.py import \
      --input data/sft/e02_v2_1/review_candidates.jsonl \
      --excel data/sft/e02_v2_1/review_e02.xlsx \
      --output data/sft/e02_v2_1/review_candidates_reviewed.jsonl

Requiere: pip install artifact_tool
No usa IA para aprobar registros; todo 'approved' exige revisor y datos válidos.
No modifica el dataset original ni divide Train/Val/Test.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path

LABELS = ("Bien", "Servicio", "Obra", "Consultoría de Obra")

TASK_SHEETS = {
    "classification_hard": "Clasificacion",
    "document_grounded_qa": "Documentos",
}

HEADER = [
    "candidate_id",
    "task",
    "etiqueta_oficial",
    "texto_fuente",
    "fuente",
    "group_id",
    "prioridad_heuristica",
    "estado_revision",
    "revisor",
    "dificultad_humana",
    "categoria_aprobada",
    "respuesta_documental",
    "cita_evidencia_literal",
    "notas_revision",
    "vigencia_verificada",
    "regimen_normativo",
    "fecha_documento",
    "notas_normativas",
]

EDIT = {
    "estado_revision",
    "revisor",
    "dificultad_humana",
    "categoria_aprobada",
    "respuesta_documental",
    "cita_evidencia_literal",
    "notas_revision",
    "vigencia_verificada",
    "regimen_normativo",
    "fecha_documento",
    "notas_normativas",
}

REQUIRED = {
    "candidate_id",
    "task",
    "etiqueta_oficial",
    "texto_fuente",
    "fuente",
    "group_id",
}


def read_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)

    out = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as stream:

        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue

            item = json.loads(line)

            if (
                not isinstance(item, dict)
                or not item.get("candidate_id")
            ):
                raise ValueError(
                    f"Registro inválido {path}:{number}"
                )

            out.append(item)

    ids = [
        r["candidate_id"]
        for r in out
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Hay candidate_id duplicados en el JSONL original"
        )

    return out


def save_jsonl(
    path: Path,
    rows: list[dict],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                ) + "\n"
            )


def cell_text(value: object) -> str:
    if value is None:
        return ""

    if isinstance(value, bool):
        return "SI" if value else "NO"

    return str(value).strip()


def no_formula(value: str) -> str:
    # Previene que Excel interprete texto fuente como fórmula.
    if value and value.lstrip().startswith(
        ("=", "+", "-", "@")
    ):
        return "'" + value

    return value


def display_fields(rec: dict) -> dict:
    raw_input = rec.get("input")

    source_input = (
        json.loads(raw_input)
        if isinstance(raw_input, str)
        else raw_input
    )

    if not isinstance(source_input, dict):
        raise ValueError(
            f"input no es objeto: {rec['candidate_id']}"
        )

    task = rec.get("task")

    if task == "classification_hard":

        text = source_input.get("title", "")

        if source_input.get("item_description"):
            text += (
                " | ITEM: "
                + str(source_input["item_description"])
            )

        label = (
            rec.get("suggested_answer") or {}
        ).get("category", "")

        source_name = (
            rec.get("source") or {}
        ).get("procedure_id", "")

    elif task == "document_grounded_qa":

        text = source_input.get("excerpt", "")
        label = ""

        source_name = (
            rec.get("source") or {}
        ).get("relative_file", "")

    else:
        raise ValueError(
            f"Tarea desconocida: {task}"
        )

    review = rec.get("review") or {}
    answer = review.get("answer") or {}
    normative = rec.get("normative_review") or {}
    selection = rec.get("selection") or {}

    return {
        "candidate_id": rec["candidate_id"],
        "task": task,
        "etiqueta_oficial": label,
        "texto_fuente": str(text),
        "fuente": str(source_name),
        "group_id": str(
            rec.get("group_id", "")
        ),
        "prioridad_heuristica": selection.get(
            "heuristic_priority", ""
        ),
        "estado_revision": review.get(
            "status", "pending"
        ),
        "revisor": review.get(
            "reviewer", ""
        ),
        "dificultad_humana": selection.get(
            "difficulty_human", ""
        ),
        "categoria_aprobada": answer.get(
            "category", ""
        ),
        "respuesta_documental": answer.get(
            "answer", ""
        ),
        "cita_evidencia_literal": answer.get(
            "evidence_quote", ""
        ),
        "notas_revision": review.get(
            "notes", ""
        ),
        "vigencia_verificada": (
            "SI" if normative.get("verified")
            else "NO"
        ),
        "regimen_normativo": (
            normative.get("regime") or ""
        ),
        "fecha_documento": (
            normative.get("document_date") or ""
        ),
        "notas_normativas": (
            normative.get("notes") or ""
        ),
    }


def export_excel(
    original: Path,
    target: Path,
) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import (
        Font, PatternFill, Alignment
    )
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.worksheet.table import Table, TableStyleInfo
    from openpyxl.utils import get_column_letter

    records = read_records(original)

    wb = Workbook()
    instructions = wb.active
    instructions.title = "Instrucciones"

    help_rows = [
        ["PERÚCOMPRAS AI — E02 / REVISIÓN HUMANA", ""],
        ["1", "Editar solo las columnas H:R."],
        ["2", "Estados: pending, approved, rejected."],
        ["3", "Para aprobar, indicar revisor y respuesta."],
        ["4", "La categoría aprobada debe coincidir con CONOSCE."],
        ["5", "La cita documental debe existir literalmente en el texto fuente."],
        ["6", "No declarar vigencia normativa sin verificación."],
        ["7", "Marcar SI únicamente tras comprobar régimen y fecha."],
        ["8", "No modificar identificadores ni columnas de origen."],
        ["9", "Guardar Excel y ejecutar el comando import."],
        ["10", "Los rechazos deben incluir una justificación."],
    ]

    for row in help_rows:
        instructions.append(row)

    instructions.column_dimensions["A"].width = 12
    instructions.column_dimensions["B"].width = 105
    instructions["A1"].font = Font(
        bold=True, color="FFFFFF"
    )
    instructions["A1"].fill = PatternFill(
        "solid", fgColor="16324F"
    )

    widths = [
        24, 24, 22, 85, 36, 42, 20,
        20, 24, 20, 28, 75, 70,
        55, 22, 36, 22, 55,
    ]

    for task, sheet_name in TASK_SHEETS.items():
        sheet = wb.create_sheet(sheet_name)

        for col, header in enumerate(HEADER, 1):
            cell = sheet.cell(1, col, header)
            cell.font = Font(
                bold=True, color="FFFFFF"
            )
            cell.fill = PatternFill(
                "solid", fgColor="16324F"
            )
            cell.alignment = Alignment(
                wrap_text=True,
                vertical="center",
            )
            sheet.column_dimensions[
                get_column_letter(col)
            ].width = widths[col - 1]

        sheet.row_dimensions[1].height = 35
        sheet.freeze_panes = "H2"

        subset = [
            record for record in records
            if record.get("task") == task
        ]

        for row_number, record in enumerate(subset, 2):
            fields = display_fields(record)

            for col_number, key in enumerate(HEADER, 1):
                value = fields.get(key, "")

                if isinstance(value, str):
                    value = no_formula(value)

                if value is None:
                    value = ""

                cell = sheet.cell(
                    row_number, col_number, value
                )

                cell.alignment = Alignment(
                    wrap_text=True,
                    vertical="top",
                )

                if col_number >= 8:
                    cell.fill = PatternFill(
                        "solid", fgColor="EAF4FD"
                    )

            sheet.row_dimensions[row_number].height = 48

        last = len(subset) + 1

        if subset:
            table = Table(
                displayName=f"Tabla{sheet_name}",
                ref=f"A1:R{last}",
            )
            table.tableStyleInfo = TableStyleInfo(
                name="TableStyleMedium2",
                showRowStripes=True,
            )
            sheet.add_table(table)

            validations = [
                (
                    "H",
                    '"pending,approved,rejected"'
                ),
                (
                    "J",
                    '"low,medium,high"'
                ),
                (
                    "O",
                    '"NO,SI"'
                ),
            ]

            if task == "classification_hard":
                validations.append(
                    (
                        "K",
                        '"Bien,Servicio,Obra,Consultoría de Obra"'
                    )
                )

            for column, formula in validations:
                dv = DataValidation(
                    type="list",
                    formula1=formula,
                    allow_blank=True,
                )
                sheet.add_data_validation(dv)
                dv.add(f"{column}2:{column}{last}")

    target.parent.mkdir(
        parents=True, exist_ok=True
    )

    wb.save(target)

    print(f"Excel creado: {target}")
    print(
        "Candidatos:",
        dict(collections.Counter(
            record["task"] for record in records
        ))
    )


def unescape_excel(value: str) -> str:
    # Elimina el apóstrofo utilizado para proteger
    # el texto fuente de la interpretación de fórmulas.
    return (
        value[1:]
        if (
            value.startswith("'")
            and value[1:].lstrip().startswith(
                ("=", "+", "-", "@")
            )
        )
        else value
    )


def import_reviews(
    original: Path,
    excel_path: Path,
    output: Path,
    report: Path,
    strict: bool,
) -> None:

    from openpyxl import load_workbook

    if original.resolve() == output.resolve():
        raise ValueError(
            "El JSONL de salida no puede sobrescribir al original"
        )

    records = read_records(original)

    by_id = {
        r["candidate_id"]: r
        for r in records
    }

    seen: dict[str, dict] = {}

    wb = load_workbook(
        excel_path,
        data_only=True,
        read_only=False
    )

    for task, sheet_name in TASK_SHEETS.items():

        if sheet_name not in wb.sheetnames:
            raise ValueError(
                f"No existe la hoja {sheet_name}"
            )

        sheet = wb[sheet_name]

        expected = [
            r for r in records
            if r["task"] == task
        ]

        end = len(expected) + 1

        values = [
            list(row)
            for row in sheet.iter_rows(
                min_row=1,
                max_row=end,
                min_col=1,
                max_col=len(HEADER),
                values_only=True,
            )
        ]

        if [
            cell_text(v)
            for v in values[0]
        ] != HEADER:

            raise ValueError(
                f"Encabezados modificados en {sheet_name}"
            )

        for excel_line, row in enumerate(
            values[1:],
            2,
        ):

            values_by_col = {
                key: cell_text(row[i])
                for i, key in enumerate(HEADER)
            }

            candidate_id = values_by_col[
                "candidate_id"
            ]

            if candidate_id not in by_id:
                raise ValueError(
                    f"ID inesperado/vacío "
                    f"{candidate_id!r} "
                    f"en {sheet_name}:{excel_line}"
                )

            if candidate_id in seen:
                raise ValueError(
                    f"ID repetido en Excel: {candidate_id}"
                )

            rec = by_id[candidate_id]

            if rec["task"] != task:
                raise ValueError(
                    f"Tarea alterada en "
                    f"{sheet_name}:{excel_line}"
                )

            expected_fields = display_fields(rec)

            for key in REQUIRED:

                got = unescape_excel(
                    values_by_col[key]
                )

                if got != cell_text(
                    expected_fields[key]
                ):
                    raise ValueError(
                        f"Campo fuente {key} modificado "
                        f"para {candidate_id}"
                    )

            seen[candidate_id] = values_by_col

    if set(seen) != set(by_id):

        missing = set(by_id) - set(seen)

        raise ValueError(
            f"Faltan {len(missing)} candidatos "
            f"en Excel; IDs: {list(missing)[:3]}"
        )

    updated = []
    issues = []
    stats = collections.Counter()

    for original_rec in records:

        rec = copy.deepcopy(original_rec)

        row = seen[
            rec["candidate_id"]
        ]

        status = (
            row["estado_revision"]
            .casefold()
            .strip()
            or "pending"
        )

        reviewer = row[
            "revisor"
        ].strip()

        difficulty = (
            row["dificultad_humana"]
            .casefold()
            .strip()
        )

        norm_flag = (
            row["vigencia_verificada"]
            .strip()
            .upper()
            or "NO"
        )

        answer = None
        problems = []

        if status not in {
            "pending",
            "approved",
            "rejected",
        }:
            problems.append(
                "estado_revision inválido"
            )

        if (
            difficulty
            and difficulty not in {
                "low",
                "medium",
                "high",
            }
        ):
            problems.append(
                "dificultad_humana inválida"
            )

        if norm_flag not in {
            "SI",
            "NO",
        }:
            problems.append(
                "vigencia_verificada requiere SI o NO"
            )

        if (
            norm_flag == "SI"
            and (
                not row["regimen_normativo"]
                or not row["fecha_documento"]
            )
        ):
            problems.append(
                "marcar SI exige régimen "
                "y fecha de documento"
            )

        if status == "approved":

            if not reviewer:
                problems.append(
                    "revisor requerido"
                )

            if rec["task"] == "classification_hard":

                category = row[
                    "categoria_aprobada"
                ]

                official = (
                    rec.get("suggested_answer") or {}
                ).get("category")

                if category not in LABELS:

                    problems.append(
                        "categoria_aprobada inválida"
                    )

                elif category != official:

                    problems.append(
                        "categoría distinta de CONOSCE: "
                        "investigar, no aprobar automáticamente"
                    )

                else:

                    answer = {
                        "category": category
                    }

            else:

                response = row[
                    "respuesta_documental"
                ]

                quote = row[
                    "cita_evidencia_literal"
                ]

                excerpt = json.loads(
                    rec["input"]
                )["excerpt"]

                if len(response) < 10:
                    problems.append(
                        "respuesta_documental requiere "
                        "10 caracteres o más"
                    )

                if (
                    len(quote) < 12
                    or quote not in excerpt
                ):
                    problems.append(
                        "cita_evidencia_literal debe "
                        "existir exactamente en el fragmento"
                    )

                if not problems:
                    answer = {
                        "answer": response,
                        "evidence_quote": quote,
                    }

        elif (
            status == "rejected"
            and not row["notas_revision"]
        ):
            problems.append(
                "rejected requiere notas_revision"
            )

        if problems:

            issues.append({
                "candidate_id": rec["candidate_id"],
                "errors": problems,
            })

            # Si strict=True, se cancela la importación.
            # En otro caso, la fila queda pendiente.
            status = "pending"
            answer = None

        rec.setdefault(
            "selection",
            {},
        )

        if difficulty:

            rec["selection"][
                "difficulty_human"
            ] = difficulty

            rec["selection"][
                "difficulty_human_verified"
            ] = bool(reviewer)

        rec["review"] = {
            "status": status,
            "answer": answer,
            "reviewer": reviewer,
            "notes": row["notas_revision"],
        }

        norm = (
            rec.get("normative_review") or {}
        )

        norm.update({
            "verified": (
                norm_flag == "SI"
                and not any(
                    "vigencia" in e
                    for e in problems
                )
            ),
            "regime": (
                row["regimen_normativo"]
                or None
            ),
            "document_date": (
                row["fecha_documento"]
                or None
            ),
            "notes": row["notas_normativas"],
        })

        rec["normative_review"] = norm

        stats[status] += 1
        updated.append(rec)

    report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.write_text(
        json.dumps(
            {
                "total": len(records),
                "status_counts": dict(stats),
                "issues_count": len(issues),
                "issues": issues,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if strict and issues:

        raise ValueError(
            f"Hay {len(issues)} revisiones inválidas; "
            f"revisar informe {report}. "
            "No se creó JSONL"
        )

    save_jsonl(
        output,
        updated,
    )

    print(
        f"JSONL revisado: {output}"
    )

    print(
        f"Informe: {report}"
    )

    print(
        "Estados:",
        dict(stats),
        "| Incidencias:",
        len(issues),
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    exp = commands.add_parser(
        "export",
        help="Exportar candidatos a Excel",
    )

    exp.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/sft/e02_v2_1/review_candidates.jsonl"
        ),
    )

    exp.add_argument(
        "--excel",
        type=Path,
        default=Path(
            "data/sft/e02_v2_1/review_e02.xlsx"
        ),
    )

    imp = commands.add_parser(
        "import",
        help="Importar decisiones desde Excel",
    )

    imp.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/sft/e02_v2_1/review_candidates.jsonl"
        ),
    )

    imp.add_argument(
        "--excel",
        type=Path,
        default=Path(
            "data/sft/e02_v2_1/review_e02.xlsx"
        ),
    )

    imp.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/sft/e02_v2_1/review_candidates_reviewed.jsonl"
        ),
    )

    imp.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/sft/e02_v2_1/review_import_report.json"
        ),
    )

    imp.add_argument(
        "--allow-invalid-as-pending",
        action="store_true",
        help=(
            "No bloquear toda importación: "
            "deja pendientes las filas inválidas"
        ),
    )

    args = parser.parse_args()

    if args.command == "export":

        export_excel(
            args.input,
            args.excel,
        )

    else:

        import_reviews(
            args.input,
            args.excel,
            args.output,
            args.report,
            strict=not args.allow_invalid_as_pending,
        )


if __name__ == "__main__":
    main()
