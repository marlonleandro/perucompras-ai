#!/usr/bin/env python3
"""PeruCompras AI E02 v2.1: selección trazable y revisión humana.

Comandos:
  generate : lee procedures.parquet y bases DOCX
  refine   : depura review_candidates.jsonl existente
  finalize : exporta SOLO ejemplos aprobados a Alpaca

Dependencias: pip install pandas pyarrow python-docx

Ninguna respuesta jurídica se genera automáticamente.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

LABELS = (
    "Bien",
    "Servicio",
    "Obra",
    "Consultoría de Obra",
)

KEYWORDS = (
    "requisito", "experiencia", "garantia", "plazo",
    "penalidad", "obligaci", "consorcio", "acredit",
    "evaluaci", "calificaci", "document", "contrat",
    "postor", "oferta", "adelanto",
)

AMBIGUOUS = (
    "mantenimiento", "instalacion", "equipamiento",
    "reparacion", "construccion", "consultor",
    "supervision", "mejoramiento", "adquisicion",
    "suministro", "montaje", "servicio", "obra",
    "provision", "implementacion",
)

FILL_PLACEHOLDER = re.compile(
    r"\[\s*(CONSIGNAR|INDICAR|INCLUIR|PRECISAR|"
    r"COMPLETAR|INSERTAR|SEÑALAR|SENALAR)\b",
    re.I,
)

REVIEW_STATES = {"pending", "approved", "rejected"}


def plain(s: str) -> str:
    s = unicodedata.normalize(
        "NFKD", str(s).casefold()
    )
    return "".join(
        ch for ch in s
        if not unicodedata.combining(ch)
    )


def whitespace(s: str) -> str:
    return " ".join(str(s).split())


def hash_of(obj) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)

    result = []

    with path.open(encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                obj = json.loads(line)

                if not isinstance(obj, dict):
                    raise ValueError(
                        f"{path}:{n}: esperado objeto JSON"
                    )

                result.append(obj)

    return result


def save_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row, ensure_ascii=False
                ) + "\n"
            )


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    path.write_text(
        json.dumps(
            obj, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )


def unpack(rec: dict) -> dict:
    value = rec.get("input", {})

    if isinstance(value, str):
        value = json.loads(value)

    if not isinstance(value, dict):
        raise ValueError(
            f"input inválido: {rec.get('candidate_id')}"
        )

    return value


def new_candidate(
    task: str,
    group_id: str,
    source: dict,
    instruction: str,
    content: dict,
    suggested: dict | None,
) -> dict:

    cid = hash_of(
        [task, group_id, content]
    )[:20]

    return {
        "candidate_id": cid,
        "task": task,
        "group_id": group_id,
        "source": source,
        "instruction": instruction,
        "input": json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "suggested_answer": suggested,
        "normative_review": {
            "verified": False,
            "regime": None,
            "document_date": None,
            "notes": "Vigencia no verificada",
        },
        "review": {
            "status": "pending",
            "answer": None,
            "reviewer": "",
            "notes": "",
        },
    }


def difficulty_score(
    content: dict,
) -> tuple[int, list[str]]:
    """
    Heurística para priorizar revisión.
    NO constituye una etiqueta validada de dificultad.
    """
    title = plain(
        " ".join(str(x) for x in content.values())
    )

    found = [
        w for w in AMBIGUOUS
        if w in title
    ]

    mixed = [
        ("suministro", "instalacion"),
        ("adquisicion", "instalacion"),
        ("mantenimiento", "reparacion"),
        ("consultor", "supervision"),
        ("obra", "servicio"),
    ]

    pairs = [
        f"{a}+{b}"
        for a, b in mixed
        if a in title and b in title
    ]

    score = min(
        20,
        len(found) * 2 + len(pairs) * 5,
    )

    return score, pairs


def norm_tokens(s: str) -> set[str]:
    return set(
        re.findall(
            r"[a-z0-9]{3,}",
            plain(s),
        )
    )


def near_duplicate(
    a: str,
    b: str,
    threshold: float,
) -> bool:

    ta = norm_tokens(a)
    tb = norm_tokens(b)

    if not ta or not tb:
        return False

    similarity = (
        len(ta & tb) / len(ta | tb)
    )

    return similarity >= threshold


def select_candidates(
    records: list[dict],
    per_label: int,
    max_docs: int,
    threshold: float,
    seed: int,
    include_templates: bool,
) -> tuple[list[dict], dict]:

    rng = random.Random(seed)
    counts = collections.Counter()

    by_label = {
        label: [] for label in LABELS
    }

    docs = []
    seen_candidates = set()
    seen_procedures = set()

    for record in records:
        cid = record.get("candidate_id")

        if not cid or cid in seen_candidates:
            counts[
                "missing_or_duplicate_candidate_id"
            ] += 1
            continue

        seen_candidates.add(cid)

        task = record.get("task")

        try:
            content = unpack(record)
        except (
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            counts["bad_input"] += 1
            continue

        record.setdefault(
            "review",
            {
                "status": "pending",
                "answer": None,
                "reviewer": "",
                "notes": "",
            },
        )

        record.setdefault(
            "normative_review",
            {
                "verified": False,
                "regime": None,
                "document_date": None,
                "notes": "Vigencia no verificada",
            },
        )

        if task == "classification_hard":
            proposed = (
                record.get("suggested_answer") or {}
            )

            label = proposed.get("category")

            if label not in LABELS:
                counts[
                    "unknown_classification_label"
                ] += 1
                continue

            group = str(
                record.get("group_id", "")
            )

            if (
                not group
                or group in seen_procedures
            ):
                counts[
                    "duplicate_procedure"
                ] += 1
                continue

            seen_procedures.add(group)

            score, pairs = difficulty_score(
                content
            )

            record["selection"] = {
                "heuristic_priority": score,
                "mixed_clues": pairs,
                "difficulty_human_verified": False,
            }

            by_label[label].append(record)

        elif task == "document_grounded_qa":
            excerpt = str(
                content.get("excerpt", "")
            )

            if not 100 <= len(excerpt) <= 1600:
                counts[
                    "document_length_filter"
                ] += 1
                continue

            is_template = bool(
                FILL_PLACEHOLDER.search(excerpt)
            )

            if (
                is_template
                and not include_templates
            ):
                counts[
                    "template_placeholder"
                ] += 1
                continue

            if not any(
                w in plain(excerpt)
                for w in KEYWORDS
            ):
                counts[
                    "not_document_relevant"
                ] += 1
                continue

            record["selection"] = {
                "template_placeholder": is_template
            }

            docs.append(record)

        else:
            counts["unknown_task"] += 1

    selected = []

    # Muestreo por categoría oficial.
    for label in LABELS:
        bucket = by_label[label]

        rng.shuffle(bucket)

        bucket.sort(
            key=lambda r: -r["selection"][
                "heuristic_priority"
            ]
        )

        chosen = bucket[:per_label]
        selected.extend(chosen)

        counts[f"available_{label}"] = len(bucket)
        counts[f"selected_{label}"] = len(chosen)

    # Eliminación de duplicados documentales.
    rng.shuffle(docs)

    docs.sort(
        key=lambda r: -len(
            norm_tokens(
                unpack(r)["excerpt"]
            )
        )
    )

    distinct = []
    seen_doc_hashes = set()

    for record in docs:
        excerpt = str(
            unpack(record)["excerpt"]
        )

        exact_hash = hash_of(
            whitespace(plain(excerpt))
        )

        if exact_hash in seen_doc_hashes:
            counts[
                "exact_document_duplicates"
            ] += 1
            continue

        if any(
            near_duplicate(
                excerpt,
                unpack(old)["excerpt"],
                threshold,
            )
            for old in distinct
        ):
            counts[
                "near_document_duplicates"
            ] += 1
            continue

        distinct.append(record)
        seen_doc_hashes.add(exact_hash)

        if len(distinct) >= max_docs:
            break

    selected.extend(distinct)

    counts["selected_document"] = len(distinct)
    counts["available_document"] = len(docs)

    return selected, dict(counts)


def generate_source(args) -> list[dict]:
    import pandas as pd
    from docx import Document

    if not args.procedures.is_file():
        raise FileNotFoundError(
            args.procedures
        )

    df = pd.read_parquet(
        args.procedures
    )

    columns = {
        re.sub(
            r"[^a-z0-9]",
            "",
            plain(c),
        ): c
        for c in df.columns
    }

    def column(*alternatives):
        for name in alternatives:
            key = re.sub(
                r"[^a-z0-9]",
                "",
                plain(name),
            )

            if key in columns:
                return columns[key]

        return None

    c_id = column(
        "procedure_id",
        "codigo_convocatoria",
        "codigoconvocatoria",
    )

    c_label = column(
        "category",
        "objeto_contractual",
        "objetocontractual",
    )

    c_title = column(
        "title",
        "description",
        "descripcion_proceso",
        "descripcionproceso",
        "objeto_del_proceso",
    )

    if not all([c_id, c_label, c_title]):
        raise ValueError(
            "Parquet sin columnas requeridas. "
            f"Columnas: {list(df.columns)}"
        )

    def cell(v):
        return (
            ""
            if pd.isna(v)
            else str(v).strip()
        )

    candidates = []
    seen = set()

    # CONOSCE: candidatos de clasificación.
    for _, row in df.iterrows():
        pid = cell(row[c_id])
        label = cell(row[c_label])
        title = cell(row[c_title])

        if (
            not pid
            or pid in seen
            or label not in LABELS
            or len(title) < 25
        ):
            continue

        seen.add(pid)

        score, _ = difficulty_score(
            {"title": title}
        )

        if score == 0:
            continue

        candidates.append(
            new_candidate(
                "classification_hard",
                "procedure:" + pid,
                {
                    "kind": "CONOSCE",
                    "file": args.procedures.as_posix(),
                    "procedure_id": pid,
                    "label_origin": "official_category",
                },
                (
                    "Clasifica el objeto contractual "
                    "solo como Bien, Servicio, Obra "
                    "o Consultoría de Obra. Devuelve "
                    "JSON con la única clave category."
                ),
                {"title": title},
                {"category": label},
            )
        )

    # Bases estándar: fragmentos documentales.
    files = (
        sorted(args.docs.rglob("*.docx"))
        if args.docs.exists()
        else []
    )

    for path in files:
        if path.name.startswith("~$"):
            continue

        try:
            doc = Document(str(path))
        except Exception as exc:
            print(
                f"[WARN] No se puede leer {path}: {exc}"
            )
            continue

        relative = path.relative_to(
            args.docs
        ).as_posix()

        chunks = [
            p.text for p in doc.paragraphs
        ]

        for table in doc.tables:
            for row in table.rows:
                chunks.append(
                    " | ".join(
                        cell.text
                        for cell in row.cells
                    )
                )

        for position, raw in enumerate(chunks):
            excerpt = whitespace(raw)

            if not 100 <= len(excerpt) <= 1600:
                continue

            if not any(
                w in plain(excerpt)
                for w in KEYWORDS
            ):
                continue

            candidates.append(
                new_candidate(
                    "document_grounded_qa",
                    "document:" + relative,
                    {
                        "kind": "bases_estandar",
                        "file": path.as_posix(),
                        "relative_file": relative,
                        "paragraph_index": position,
                        "excerpt_sha256": hashlib.sha256(
                            excerpt.encode("utf-8")
                        ).hexdigest(),
                    },
                    (
                        "A partir del fragmento, explica "
                        "una condición o requisito verificable. "
                        "Responde solo JSON con answer y "
                        "evidence_quote; no afirmes vigencia "
                        "normativa que no esté verificada."
                    ),
                    {
                        "source_document": relative,
                        "excerpt": excerpt,
                    },
                    None,
                )
            )

    return candidates


def run_selection(args) -> None:
    if args.command == "generate":
        rows = generate_source(args)
    else:
        rows = read_jsonl(args.input)

    selected, statistics = select_candidates(
        rows,
        args.per_label,
        args.max_document,
        args.near_threshold,
        args.seed,
        args.include_templates,
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_jsonl(
        args.output / "review_candidates.jsonl",
        selected,
    )

    save_json(
        args.output / "generation_manifest.json",
        {
            "stage": "candidate_generation_only",
            "version": "2.1",
            "seed": args.seed,
            "input_candidates": len(rows),
            "selected_candidates": len(selected),
            "task_counts": dict(
                collections.Counter(
                    r["task"]
                    for r in selected
                )
            ),
            "selection_stats": statistics,
            "near_duplicate_jaccard_threshold": (
                args.near_threshold
            ),
            "warning": (
                "NO ENTRENAR. Las etiquetas y respuestas "
                "requieren revisión humana; vigencia "
                "normativa no verificada."
            ),
        },
    )

    print(
        json.dumps(
            {
                "selected": len(selected),
                "stats": statistics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def validate_review(
    record: dict,
) -> tuple[bool, str]:

    review = record.get("review") or {}

    status = review.get(
        "status", "pending"
    )

    if status not in REVIEW_STATES:
        return False, "invalid_review_status"

    if status != "approved":
        return False, "not_approved"

    if not str(
        review.get("reviewer") or ""
    ).strip():
        return False, "missing_reviewer"

    answer = review.get("answer")

    if not isinstance(answer, dict):
        return False, "answer_not_json_object"

    if record["task"] == "classification_hard":
        if (
            set(answer) != {"category"}
            or answer["category"] not in LABELS
        ):
            return (
                False,
                "invalid_classification_schema",
            )

        if answer != record.get(
            "suggested_answer"
        ):
            return (
                False,
                "disagrees_with_official_label_"
                "manual_investigation_needed",
            )

        return True, "ok"

    if record["task"] == "document_grounded_qa":
        if set(answer) != {
            "answer",
            "evidence_quote",
        }:
            return (
                False,
                "invalid_document_schema",
            )

        if (
            not isinstance(
                answer["answer"], str
            )
            or len(
                answer["answer"].strip()
            ) < 10
        ):
            return False, "invalid_answer"

        quote = answer["evidence_quote"]

        if (
            not isinstance(quote, str)
            or len(quote.strip()) < 12
        ):
            return False, "invalid_evidence"

        excerpt = unpack(record).get(
            "excerpt", ""
        )

        if quote not in excerpt:
            return (
                False,
                "quote_not_exact_substring",
            )

        return True, "ok"

    return False, "unknown_task"


def split_for_group(
    group: str,
    seed: int,
    train_percent: int,
    val_percent: int,
) -> str:

    bucket = int(
        hash_of([seed, group])[:12],
        16,
    ) % 100

    if bucket < train_percent:
        return "train"

    if bucket < train_percent + val_percent:
        return "val"

    return "test"


def finalize(args) -> None:
    rows = read_jsonl(args.input)

    seen_ids = set()
    approved = []
    issues = []

    for rec in rows:
        cid = rec.get("candidate_id")

        if not cid or cid in seen_ids:
            raise ValueError(
                f"ID duplicado o ausente: {cid}"
            )

        seen_ids.add(cid)

        good, reason = validate_review(rec)

        if good:
            approved.append(rec)
        else:
            issues.append({
                "candidate_id": cid,
                "reason": reason,
            })

    unique = []
    seen_input = set()

    for rec in approved:
        content = unpack(rec)

        key = hash_of([
            rec["task"],
            content.get("excerpt", content),
        ])

        if key in seen_input:
            issues.append({
                "candidate_id": rec["candidate_id"],
                "reason": "duplicate_approved_input",
            })
            continue

        seen_input.add(key)
        unique.append(rec)

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    groups = {}

    for rec in unique:
        group = rec["group_id"]

        split = split_for_group(
            group,
            args.seed,
            args.train_pct,
            args.val_pct,
        )

        if (
            group in groups
            and groups[group] != split
        ):
            raise AssertionError(
                "Fuga entre splits"
            )

        groups[group] = split

        splits[split].append({
            "instruction": rec["instruction"],
            "input": rec["input"],
            "output": json.dumps(
                rec["review"]["answer"],
                ensure_ascii=False,
            ),
        })

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split, examples in splits.items():
        save_jsonl(
            args.output / f"{split}.jsonl",
            examples,
        )

    save_jsonl(
        args.output / "review_issues.jsonl",
        issues,
    )

    save_json(
        args.output / "dataset_manifest.json",
        {
            "version": "2.1",
            "stage": "human_approved_only",
            "input_candidates": len(rows),
            "approved_exported": len(unique),
            "excluded": len(issues),
            "split_counts": {
                k: len(v)
                for k, v in splits.items()
            },
            "approved_task_counts": dict(
                collections.Counter(
                    r["task"]
                    for r in unique
                )
            ),
            "seed": args.seed,
            "train_pct": args.train_pct,
            "val_pct": args.val_pct,
            "warnings": [
                (
                    "Revisión humana no demuestra "
                    "vigencia normativa."
                ),
                (
                    "Antes de entrenar, auditar similitud "
                    "entre ejemplos y cobertura de "
                    "clases por split."
                ),
            ],
        },
    )

    print(
        json.dumps(
            {
                "approved_exported": len(unique),
                "split_counts": {
                    k: len(v)
                    for k, v in splits.items()
                },
                "excluded": len(issues),
            },
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__
    )

    cmds = p.add_subparsers(
        dest="command",
        required=True,
    )

    for action in ("generate", "refine"):
        sub = cmds.add_parser(action)

        if action == "generate":
            sub.add_argument(
                "--procedures",
                type=Path,
                default=Path(
                    "data/processed/procedures.parquet"
                ),
            )

            sub.add_argument(
                "--docs",
                type=Path,
                default=Path("data/raw"),
            )
        else:
            sub.add_argument(
                "--input",
                type=Path,
                default=Path(
                    "data/sft/e02/review_candidates.jsonl"
                ),
            )

        sub.add_argument(
            "--output",
            type=Path,
            default=Path("data/sft/e02_v2_1"),
        )

        sub.add_argument(
            "--per-label",
            type=int,
            default=40,
        )

        sub.add_argument(
            "--max-document",
            type=int,
            default=120,
        )

        sub.add_argument(
            "--near-threshold",
            type=float,
            default=0.85,
        )

        sub.add_argument(
            "--include-templates",
            action="store_true",
        )

        sub.add_argument(
            "--seed",
            type=int,
            default=42,
        )

    sub = cmds.add_parser("finalize")

    sub.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/sft/e02_v2_1/review_candidates.jsonl"
        ),
    )

    sub.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/sft/e02_v2_1/approved"
        ),
    )

    sub.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    sub.add_argument(
        "--train-pct",
        type=int,
        default=80,
    )

    sub.add_argument(
        "--val-pct",
        type=int,
        default=10,
    )

    args = p.parse_args()

    if args.command in ("generate", "refine"):
        if (
            args.per_label < 0
            or args.max_document < 0
            or not (
                0 < args.near_threshold <= 1
            )
        ):
            p.error(
                "Parámetros de selección inválidos"
            )

        run_selection(args)

    else:
        if (
            args.train_pct <= 0
            or args.val_pct <= 0
            or args.train_pct + args.val_pct >= 100
        ):
            p.error(
                "Porcentajes de splits inválidos"
            )

        finalize(args)


if __name__ == "__main__":
    main()
