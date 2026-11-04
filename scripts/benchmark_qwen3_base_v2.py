
#!/usr/bin/env python3
"""PeruCompras AI: baseline v2, contrato JSON explicito y evaluacion por tarea.

Uso:
python scripts/benchmark_qwen3_base_v2.py \
    --input data/sft/pilot/val.jsonl \
    --output data/reports/benchmark_base_val_v2 \
    --quantization 4bit

No afina el modelo; no modifica ejemplos ni las etiquetas de referencia.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import statistics
import time
from pathlib import Path


LABELS = (
    "Bien",
    "Servicio",
    "Obra",
    "Consultoría de Obra"
)

FIELD_MAP = {
    "descripcion_del_item": "description",
    "entidad": "entity_name",
    "fecha_convocatoria": "publication_date",
    "moneda_item": "item_currency",
    "moneda_proceso": "procedure_currency",
    "monto_referencial_item": "item_estimated_value",
    "monto_referencial_proceso": "procedure_estimated_value",
    "objeto_contractual": "category",
    "objeto_del_proceso": "title",
    "tipo_de_seleccion": "procurement_type",
}

OUTPUT_FIELDS = tuple(FIELD_MAP.values())

PROMPT_VERSION = "baseline_v2_schema_contract_1"

SYSTEM = (
    "Eres un analista de datos de contrataciones públicas del Perú. "
    "Cumple estrictamente el contrato JSON proporcionado. "
    "No añadas Markdown, explicaciones ni claves adicionales. "
    "Usa únicamente los datos del mensaje del usuario. "
    "No deduzcas normas ni información ausente."
)


def task_from_instruction(instruction: str) -> str:

    if instruction.startswith(
        "Clasifica el objeto contractual"
    ):
        return "classification"

    if instruction.startswith(
        "Extrae los atributos verificables"
    ):
        return "extraction"

    raise ValueError(
        f"Instruccion desconocida: {instruction[:90]!r}"
    )


def build_messages(rec: dict, task: str) -> list[dict]:
    """
    Construye mensajes sin utilizar la respuesta esperada.
    Evita filtración de etiquetas desde el benchmark.
    """

    source = json.loads(rec["input"])

    if not isinstance(source, dict):
        raise ValueError(
            "La entrada debe ser un objeto JSON"
        )

    if task == "classification":

        contract = (
            'Devuelve exactamente UN objeto JSON con UNA sola clave "category".\n'
            'El valor debe ser exactamente una de estas cuatro cadenas '
            '(respetando mayúsculas, acentos y singular):\n'
            '"Bien", "Servicio", "Obra", "Consultoría de Obra".\n'
            'Ejemplo de FORMA, NO de respuesta: '
            '{"category": "<una de las cuatro etiquetas>"}.\n'
            'Clasifica según el objeto de contratación descrito; '
            'no crees categorías nuevas.'
        )

    else:

        mapping = json.dumps(
            FIELD_MAP,
            ensure_ascii=False,
            indent=2
        )

        contract = (
            "Transforma el registro de entrada a un JSON "
            "con exactamente estas diez claves: "
            + ", ".join(
                f'"{x}"' for x in OUTPUT_FIELDS
            )
            + ".\n"
            "Copia cada valor de la clave fuente a la clave "
            "de destino indicada abajo. "
            "Conserva el tipo JSON del valor de entrada "
            "cuando corresponda. "
            "No alteres espacios, guiones, caracteres, "
            "decimales, mayúsculas ni errores ortográficos. "
            "No conviertas importes a números ni cambies "
            "su representación como texto. "
            "Si la clave fuente no existe, usa null; "
            "no inventes datos.\n"
            "Mapa exacto de claves (origen -> destino):\n"
            + mapping
        )

    user = (
        contract
        + "\n\nREGISTRO FUENTE (JSON):\n"
        + json.dumps(source, ensure_ascii=False)
    )

    return [
        {
            "role": "system",
            "content": SYSTEM
        },
        {
            "role": "user",
            "content": user
        }
    ]


def parse_json(text: str) -> tuple[dict | None, bool]:
    """
    Devuelve (objeto_parseado, json_estricto).

    No recupera JSON dentro de respuestas con texto adicional.
    """

    s = text.strip()

    try:
        parsed = json.loads(s)

        if isinstance(parsed, dict):
            return parsed, True

    except (json.JSONDecodeError, TypeError):
        pass

    return None, False


def expected_from_record(rec: dict, task: str) -> dict:

    expected = json.loads(rec["output"])

    if not isinstance(expected, dict):
        raise ValueError(
            "Salida esperada no es un objeto JSON"
        )

    if task == "classification":

        if (
            set(expected) != {"category"}
            or expected["category"] not in LABELS
        ):
            raise ValueError(
                "Etiqueta de clasificacion fuera de contrato"
            )

    else:

        if set(expected) != set(OUTPUT_FIELDS):
            raise ValueError(
                "Salida de extraccion no coincide "
                "con el contrato de 10 campos"
            )

    return expected


def percentile(
    values: list[float],
    pct: float
) -> float | None:

    if not values:
        return None

    data = sorted(values)

    index = (len(data) - 1) * pct

    low = math.floor(index)
    high = math.ceil(index)

    return round(
        data[low]
        + (data[high] - data[low]) * (index - low),
        3
    )


def summarize(rows: list[dict]) -> dict:

    groups = collections.defaultdict(list)

    for row in rows:
        groups[row["task"]].append(row)

    output = {
        "n": len(rows),
        "tasks": {}
    }

    for task, group in groups.items():

        n = len(group)

        d = {
            "count": n,

            "strict_json_valid_rate": round(
                sum(
                    r["json_valid"]
                    for r in group
                ) / n,
                4
            ),

            "schema_compliance_rate": round(
                sum(
                    r["schema_compliant"]
                    for r in group
                ) / n,
                4
            ),

            "truncation_rate": round(
                sum(
                    r["truncated"]
                    for r in group
                ) / n,
                4
            ),

            "mean_latency_seconds": round(
                statistics.mean(
                    r["latency_seconds"]
                    for r in group
                ),
                3
            ),

            "p95_latency_seconds": percentile(
                [
                    r["latency_seconds"]
                    for r in group
                ],
                0.95
            )
        }

        if task == "classification":

            d["accuracy"] = round(
                sum(
                    r["correct"]
                    for r in group
                ) / n,
                4
            )

            f1s = []

            for label in LABELS:

                tp = sum(
                    r["expected"]["category"] == label
                    and (r["parsed"] or {}).get("category") == label
                    for r in group
                )

                fp = sum(
                    r["expected"]["category"] != label
                    and (r["parsed"] or {}).get("category") == label
                    for r in group
                )

                fn = sum(
                    r["expected"]["category"] == label
                    and (r["parsed"] or {}).get("category") != label
                    for r in group
                )

                denominator = 2 * tp + fp + fn

                f1s.append(
                    2 * tp / denominator
                    if denominator else 0.0
                )

            d["macro_f1_all_four_labels"] = round(
                statistics.mean(f1s),
                4
            )

            d["labels"] = list(LABELS)

            d["confusion"] = {
                label: dict(
                    collections.Counter(
                        (r["parsed"] or {}).get(
                            "category",
                            "<invalid>"
                        )
                        for r in group
                        if r["expected"]["category"] == label
                    )
                )
                for label in LABELS
            }

        else:

            total = sum(
                len(r["expected"])
                for r in group
            )

            correct_fields = sum(
                sum(
                    (r["parsed"] or {}).get(
                        k,
                        object()
                    ) == v
                    for k, v in r["expected"].items()
                )
                for r in group
            )

            d["field_exact_match"] = (
                round(correct_fields / total, 4)
                if total else None
            )

            d["record_exact_match"] = round(
                sum(
                    r["parsed"] == r["expected"]
                    for r in group
                ) / n,
                4
            )

            d["correct_fields"] = correct_fields
            d["total_fields"] = total

            d["note"] = (
                "Transformacion estructurada de campos "
                "proporcionados; no mide extraccion "
                "desde documentos externos."
            )

        output["tasks"][task] = d

    return output


def read_dataset(path: Path, limit: int):

    records = []

    with path.open(
        "r",
        encoding="utf-8-sig"
    ) as handle:

        for line_number, line in enumerate(
            handle,
            1
        ):

            if not line.strip():
                continue

            rec = json.loads(line)

            task = task_from_instruction(
                rec["instruction"]
            )

            gold = expected_from_record(
                rec,
                task
            )

            messages = build_messages(
                rec,
                task
            )

            records.append(
                (
                    line_number,
                    rec,
                    task,
                    gold,
                    messages
                )
            )

            if limit and len(records) >= limit:
                break

    if not records:
        raise ValueError("Dataset vacio")

    return records


def main():

    p = argparse.ArgumentParser(
        description=__doc__
    )

    p.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/sft/pilot/val.jsonl"
        )
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/reports/benchmark_base_val_v2"
        )
    )

    p.add_argument(
        "--model",
        default="Qwen/Qwen3-8B"
    )

    p.add_argument(
        "--quantization",
        choices=("none", "4bit"),
        default="4bit"
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=768
    )

    p.add_argument(
        "--max-input-tokens",
        type=int,
        default=2048
    )

    p.add_argument(
        "--limit",
        type=int,
        default=0
    )

    p.add_argument(
        "--allow-test",
        action="store_true"
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Verifica dataset y muestra contrato "
            "sin cargar GPU ni escribir predicciones"
        )
    )

    args = p.parse_args()

    if (
        args.input.name.lower() == "test.jsonl"
        and not args.allow_test
    ):
        p.error(
            "El conjunto Test esta reservado; "
            "use --allow-test solo al final del estudio"
        )

    if (
        args.max_new_tokens < 16
        or args.max_input_tokens < 128
        or args.limit < 0
    ):
        p.error(
            "Limites de tokens o limit invalidos"
        )

    if (
        args.output.resolve()
        == args.input.parent.resolve()
    ):
        p.error(
            "La carpeta de salida no puede ser "
            "la misma que contiene el dataset"
        )

    records = read_dataset(
        args.input,
        args.limit
    )

    if args.dry_run:

        print(
            f"OK: {len(records)} ejemplos validados; "
            f"prompt version: {PROMPT_VERSION}"
        )

        for task in (
            "classification",
            "extraction"
        ):

            example = next(
                (
                    r for r in records
                    if r[2] == task
                ),
                None
            )

            if example:

                print(
                    f"\n--- {task}: system ---\n"
                    f"{example[4][0]['content']}"
                )

                print(
                    f"\n--- {task}: user "
                    f"(primeros 1600 caracteres) ---\n"
                    f"{example[4][1]['content'][:1600]}"
                )

        return

    import torch
    import transformers

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )

    if (
        args.quantization == "4bit"
        and not torch.cuda.is_available()
    ):
        p.error(
            "4bit requiere PyTorch con CUDA funcional"
        )

    print(
        f"Cargando {args.model} "
        f"({args.quantization}), "
        f"{len(records)} registros...",
        flush=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    model_kwargs = {
        "device_map": "auto",
        "dtype": "auto"
    }

    if args.quantization == "4bit":

        model_kwargs["quantization_config"] = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs
    )

    model.eval()

    device = (
        model.get_input_embeddings().weight.device
    )

    print(
        f"Modelo cargado en {device}",
        flush=True
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    predictions_file = (
        args.output / "predictions.jsonl"
    )

    with predictions_file.open(
        "w",
        encoding="utf-8"
    ) as out:

        for i, (
            line_number,
            rec,
            task,
            gold,
            messages
        ) in enumerate(records, 1):

            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt"
            )

            # BatchEncoding -> tensores PyTorch
            model_inputs = {
                k: v.to(device)
                for k, v in encoded.items()
            }

            prompt_length = int(
                model_inputs["input_ids"].shape[-1]
            )

            if (
                prompt_length
                > args.max_input_tokens
            ):
                raise ValueError(
                    f"Linea {line_number}: "
                    f"prompt de {prompt_length} tokens "
                    f"excede --max-input-tokens="
                    f"{args.max_input_tokens}. "
                    "No se trunca."
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.inference_mode():

                generated = model.generate(
                    **model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter() - start
            )

            # Excluir tokens del prompt.
            new_ids = generated[
                0,
                prompt_length:
            ]

            response = tokenizer.decode(
                new_ids,
                skip_special_tokens=True
            ).strip()

            parsed, valid = parse_json(
                response
            )

            expected_keys = (
                {"category"}
                if task == "classification"
                else set(OUTPUT_FIELDS)
            )

            schema_compliant = bool(
                valid
                and set(parsed) == expected_keys
                and (
                    task != "classification"
                    or parsed["category"] in LABELS
                )
            )

            # Comprobar si terminó por token especial.
            end_ids = set(
                tokenizer.all_special_ids
            )

            stopped_on_special = bool(
                new_ids.numel() > 0
                and int(new_ids[-1]) in end_ids
            )

            truncated = bool(
                new_ids.numel()
                >= args.max_new_tokens
                and not stopped_on_special
            )

            row = {
                "line": line_number,
                "task": task,
                "expected": gold,
                "parsed": parsed,
                "raw_response": response,
                "json_valid": valid,
                "schema_compliant": schema_compliant,
                "correct": (
                    bool(parsed == gold)
                    if task == "classification"
                    else None
                ),
                "truncated": truncated,
                "stopped_on_special": stopped_on_special,
                "latency_seconds": round(
                    elapsed,
                    3
                ),
                "prompt_tokens": prompt_length,
                "generated_tokens": int(
                    new_ids.numel()
                )
            }

            results.append(row)

            out.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )

            out.flush()

            print(
                f"[{i}/{len(records)}] "
                f"{task} "
                f"json={valid} "
                f"schema={schema_compliant} "
                f"truncated={truncated} "
                f"{elapsed:.2f}s",
                flush=True
            )

    summary = summarize(results)

    summary["run"] = {
        "model": args.model,
        "input": str(args.input),
        "input_sha256": hashlib.sha256(
            args.input.read_bytes()
        ).hexdigest(),
        "quantization": args.quantization,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "n": len(records),
        "date_utc": dt.datetime.now(
            dt.timezone.utc
        ).isoformat(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "enable_thinking": False,
        "do_sample": False,
        "prompt_version": PROMPT_VERSION,
        "json_schema_version": (
            "perucompras-extraction-10-fields-v1"
        ),
        "notice": (
            "No comparable directamente con baseline_v1: "
            "el prompt fue modificado y el presupuesto "
            "de tokens aumento a 768."
        )
    }

    summary_path = (
        args.output / "benchmark_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        )
    )

    print(
        "Archivos guardados en",
        args.output
    )


if __name__ == "__main__":
    main()
