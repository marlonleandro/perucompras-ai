#!/usr/bin/env python3
"""
PeruCompras AI - E02
Dataset especializado con revisión humana.

Dependencias:
    pip install pandas pyarrow python-docx

Etapas:
    1. generate: genera candidatos trazables.
    2. finalize: exporta únicamente ejemplos aprobados.

Los candidatos generados NO deben utilizarse directamente
para entrenar un modelo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys

from collections import Counter
from pathlib import Path


LABELS = (
    "Bien",
    "Servicio",
    "Obra",
    "Consultoría de Obra",
)

KEYWORDS = (
    "requisito",
    "experiencia",
    "garantía",
    "plazo",
    "penalidad",
    "obligaci",
    "consorcio",
    "acredit",
    "evaluaci",
    "calificaci",
    "document",
    "contrat",
    "postor",
    "oferta",
)

AMBIGUOUS = (
    "mantenimiento",
    "instalaci",
    "equipamiento",
    "reparaci",
    "construcci",
    "consultor",
    "supervisi",
    "mejoramiento",
    "adquisici",
    "suministro",
    "montaje",
)


def stable_hash(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def dumps(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
    )


def read_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open(encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: {exc}"
                ) from exc

    return records


def write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                ) + "\n"
            )


def first_column(
    columns: list[str],
    aliases: tuple[str, ...],
) -> str | None:
    normalized = {
        re.sub(
            r"[^a-z0-9]",
            "",
            column.casefold(),
        ): column
        for column in columns
    }

    for alias in aliases:
        key = re.sub(
            r"[^a-z0-9]",
            "",
            alias.casefold(),
        )

        if key in normalized:
            return normalized[key]

    return None


def str_val(value: object) -> str:
    if value is None:
        return ""

    try:
        import pandas as pd

        if pd.isna(value):
            return ""

    except (TypeError, ValueError, ImportError):
        pass

    return str(value).strip()


def make_candidate(
    task: str,
    group_id: str,
    source: dict,
    instruction: str,
    content: dict,
    expected: dict | None,
) -> dict:
    fingerprint = stable_hash(
        dumps([task, group_id, content])
    )

    return {
        "candidate_id": fingerprint[:20],
        "task": task,
        "group_id": group_id,
        "source": source,
        "instruction": instruction,
        "input": dumps(content),
        "suggested_answer": expected,
        "review": {
            "status": "pending",
            "answer": None,
            "reviewer": "",
            "notes": "",
        },
    }


def generate_classification(
    path: Path,
    limit: int,
    rng: random.Random,
) -> list[dict]:
    import pandas as pd

    if not path.is_file():
        print(
            f"[INFO] No existe {path}. "
            "Se omite clasificación.",
            file=sys.stderr,
        )
        return []

    df = pd.read_parquet(path)

    columns = list(df.columns)

    aliases = {
        "category": (
            "category",
            "objeto_contractual",
            "objetocontractual",
            "contractual_object",
            "object_type",
        ),
        "title": (
            "title",
            "description",
            "descripcion_proceso",
            "descripcionproceso",
            "objeto_del_proceso",
            "process_description",
        ),
        "id": (
            "procedure_id",
            "id_procedure",
            "codigo_convocatoria",
            "codigoconvocatoria",
            "process_id",
        ),
        "item": (
            "item_description",
            "descripcion_item",
            "descripcionitem",
        ),
    }

    cols = {
        key: first_column(columns, options)
        for key, options in aliases.items()
    }

    missing = [
        key
        for key in ("category", "title", "id")
        if not cols[key]
    ]

    if missing:
        raise ValueError(
            f"Columnas faltantes: {missing}. "
            f"Columnas disponibles: {columns}"
        )

    choices = []
    seen_ids = set()

    for row in df.to_dict("records"):
        label = str_val(
            row[cols["category"]]
        )

        title = str_val(
            row[cols["title"]]
        )

        procedure_id = str_val(
            row[cols["id"]]
        )

        item = (
            str_val(row[cols["item"]])
            if cols["item"]
            else ""
        )

        if (
            label not in LABELS
            or len(title) < 25
            or not procedure_id
            or procedure_id in seen_ids
        ):
            continue

        seen_ids.add(procedure_id)

        combined = (
            title + " " + item
        ).casefold()

        if not any(
            keyword in combined
            for keyword in AMBIGUOUS
        ):
            continue

        choices.append(
            (procedure_id, title, item, label)
        )

    rng.shuffle(choices)

    candidates = []

    for procedure_id, title, item, label in choices[:limit]:
        content = {
            "title": title,
        }

        if item:
            content["item_description"] = item

        candidates.append(
            make_candidate(
                task="classification_hard",
                group_id="procedure:" + procedure_id,
                source={
                    "kind": "CONOSCE",
                    "file": path.as_posix(),
                    "procedure_id": procedure_id,
                    "label_origin": "official_category",
                },
                instruction=(
                    "Clasifica el objeto contractual "
                    "únicamente como Bien, Servicio, "
                    "Obra o Consultoría de Obra. "
                    "Devuelve solo JSON con la clave "
                    "category. Usa la descripción "
                    "y no inventes datos."
                ),
                content=content,
                expected={
                    "category": label,
                },
            )
        )

    return candidates


def normalize_paragraph(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def generate_documents(
    root: Path,
    limit: int,
    rng: random.Random,
) -> list[dict]:
    from docx import Document

    files = (
        sorted(
            path
            for path in root.rglob("*.docx")
            if not path.name.startswith("~$")
        )
        if root.exists()
        else []
    )

    by_document = {}

    for path in files:
        try:
            document = Document(str(path))

        except Exception as exc:
            print(
                f"[WARN] No se pudo leer {path}: {exc}",
                file=sys.stderr,
            )
            continue

        paragraphs = [
            normalize_paragraph(p.text)
            for p in document.paragraphs
        ]

        # Incluir contenido de tablas.
        for table in document.tables:
            for row in table.rows:
                paragraphs.append(
                    " | ".join(
                        normalize_paragraph(cell.text)
                        for cell in row.cells
                    )
                )

        valid = [
            (index, text)
            for index, text in enumerate(paragraphs)
            if (
                100 <= len(text) <= 1600
                and any(
                    keyword in text.casefold()
                    for keyword in KEYWORDS
                )
            )
        ]

        seen = set()
        document_candidates = []

        for paragraph_index, excerpt in valid:
            excerpt_hash = stable_hash(excerpt)

            if excerpt_hash in seen:
                continue

            seen.add(excerpt_hash)

            relative_path = (
                path.relative_to(root).as_posix()
            )

            candidate = make_candidate(
                task="document_grounded_qa",
                group_id="document:" + relative_path,
                source={
                    "kind": "bases_estandar",
                    "file": path.as_posix(),
                    "relative_file": relative_path,
                    "paragraph_index": paragraph_index,
                    "excerpt_sha256": excerpt_hash,
                },
                instruction=(
                    "Según el fragmento de una base estándar, "
                    "identifica una condición, requisito u "
                    "obligación verificable y explica qué "
                    "establece. Responde JSON con las claves "
                    "answer y evidence_quote. Si el fragmento "
                    "no contiene información suficiente, "
                    "el candidato debe rechazarse en revisión."
                ),
                content={
                    "source_document": relative_path,
                    "excerpt": excerpt,
                },
                expected=None,
            )

            document_candidates.append(candidate)

        rng.shuffle(document_candidates)

        by_document[relative_path] = document_candidates

    # Selección equilibrada entre documentos.
    names = list(by_document)
    rng.shuffle(names)

    results = []
    depth = 0

    while len(results) < limit:
        added = False

        for name in names:
            if depth < len(by_document[name]):
                results.append(
                    by_document[name][depth]
                )

                added = True

                if len(results) >= limit:
                    break

        if not added:
            break

        depth += 1

    return results


def generate(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    candidates = generate_classification(
        args.procedures,
        args.max_classification,
        rng,
    )

    candidates += generate_documents(
        args.docs,
        args.max_document,
        rng,
    )

    if not candidates:
        raise ValueError(
            "No se generaron candidatos. "
            "Comprueba las rutas de origen."
        )

    identifiers = [
        record["candidate_id"]
        for record in candidates
    ]

    if len(identifiers) != len(set(identifiers)):
        raise ValueError(
            "Hay identificadores duplicados."
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_path = (
        args.output / "review_candidates.jsonl"
    )

    write_jsonl(
        review_path,
        candidates,
    )

    manifest = {
        "stage": "candidate_generation_only",
        "warning": (
            "NO ENTRENAR directamente con "
            "review_candidates.jsonl. "
            "Falta revisión humana."
        ),
        "total": len(candidates),
        "task_counts": dict(
            Counter(
                record["task"]
                for record in candidates
            )
        ),
        "seed": args.seed,
        "review_file": review_path.as_posix(),
    }

    (
        args.output / "generation_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
    )


def check_review(
    record: dict,
) -> tuple[bool, str]:
    review = record.get("review") or {}

    if review.get("status") != "approved":
        return False, "not_approved"

    answer = review.get("answer")

    if not isinstance(answer, dict):
        return False, "answer_must_be_json_object"

    if not str(
        review.get("reviewer", "")
    ).strip():
        return False, "missing_reviewer"

    if record["task"] == "classification_hard":
        if (
            set(answer) != {"category"}
            or answer["category"] not in LABELS
        ):
            return False, "invalid_classification_schema"

        if answer != record.get("suggested_answer"):
            return (
                False,
                "different_from_official_label"
                "_reject_or_investigate",
            )

    elif record["task"] == "document_grounded_qa":
        if set(answer) != {
            "answer",
            "evidence_quote",
        }:
            return False, "invalid_document_schema"

        text = answer["answer"]
        quote = answer["evidence_quote"]

        if (
            not isinstance(text, str)
            or len(text.strip()) < 10
        ):
            return False, "answer_too_short"

        if (
            not isinstance(quote, str)
            or len(quote.strip()) < 12
        ):
            return False, "evidence_quote_too_short"

        excerpt = json.loads(
            record["input"]
        )["excerpt"]

        if quote.strip() not in excerpt:
            return False, "evidence_not_exact_substring"

    else:
        return False, "unknown_task"

    return True, "ok"


def split_group(
    group_id: str,
    seed: int,
    train: int,
    val: int,
) -> str:
    bucket = int(
        stable_hash(
            f"{seed}|{group_id}"
        )[:12],
        16,
    ) % 100

    if bucket < train:
        return "train"

    if bucket < train + val:
        return "val"

    return "test"


def finalize(args: argparse.Namespace) -> None:
    if (
        args.train_pct <= 0
        or args.val_pct <= 0
        or args.train_pct + args.val_pct >= 100
    ):
        raise ValueError(
            "Porcentajes inválidos para Train/Val/Test"
        )

    records = read_jsonl(args.input)

    approved = []
    issues = []
    seen = set()

    for record in records:
        candidate_id = record.get("candidate_id")

        if (
            not candidate_id
            or candidate_id in seen
        ):
            raise ValueError(
                f"ID faltante o duplicado: {candidate_id}"
            )

        seen.add(candidate_id)

        ok, reason = check_review(record)

        if ok:
            approved.append(record)
        else:
            issues.append({
                "candidate_id": candidate_id,
                "reason": reason,
            })

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    source_groups = {}

    for record in approved:
        group_id = record["group_id"]

        split = split_group(
            group_id,
            args.seed,
            args.train_pct,
            args.val_pct,
        )

        if (
            group_id in source_groups
            and source_groups[group_id] != split
        ):
            raise AssertionError(
                "Fuga de datos entre particiones"
            )

        source_groups[group_id] = split

        example = {
            "instruction": record["instruction"],
            "input": record["input"],
            "output": json.dumps(
                record["review"]["answer"],
                ensure_ascii=False,
            ),
        }

        splits[split].append(example)

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_name, examples in splits.items():
        write_jsonl(
            args.output / f"{split_name}.jsonl",
            examples,
        )

    write_jsonl(
        args.output / "review_issues.jsonl",
        issues,
    )

    manifest = {
        "stage": "human_approved_only",
        "total_candidates": len(records),
        "approved": len(approved),
        "excluded": len(issues),
        "split_counts": {
            key: len(value)
            for key, value in splits.items()
        },
        "approved_tasks": dict(
            Counter(
                record["task"]
                for record in approved
            )
        ),
        "source_groups": len(source_groups),
        "seed": args.seed,
        "percentages": {
            "train": args.train_pct,
            "val": args.val_pct,
            "test": (
                100
                - args.train_pct
                - args.val_pct
            ),
        },
        "note": (
            "División por procedimiento o documento. "
            "La coincidencia literal de evidencia "
            "no garantiza suficiencia jurídica."
        ),
    }

    (
        args.output / "dataset_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    gen = subparsers.add_parser(
        "generate",
        help="Generar candidatos para revisión",
    )

    gen.add_argument(
        "--procedures",
        type=Path,
        default=Path(
            "data/processed/procedures.parquet"
        ),
    )

    gen.add_argument(
        "--docs",
        type=Path,
        default=Path("data/raw"),
    )

    gen.add_argument(
        "--output",
        type=Path,
        default=Path("data/sft/e02"),
    )

    gen.add_argument(
        "--max-classification",
        type=int,
        default=160,
    )

    gen.add_argument(
        "--max-document",
        type=int,
        default=120,
    )

    gen.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    fin = subparsers.add_parser(
        "finalize",
        help="Exportar únicamente ejemplos aprobados",
    )

    fin.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/sft/e02/review_candidates.jsonl"
        ),
    )

    fin.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/sft/e02/approved"
        ),
    )

    fin.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    fin.add_argument(
        "--train-pct",
        type=int,
        default=80,
    )

    fin.add_argument(
        "--val-pct",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if args.command == "generate":
        if (
            args.max_classification < 0
            or args.max_document < 0
        ):
            parser.error(
                "Los límites no pueden ser negativos"
            )

        generate(args)

    else:
        finalize(args)


if __name__ == "__main__":
    main()
