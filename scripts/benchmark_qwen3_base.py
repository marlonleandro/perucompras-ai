#!/usr/bin/env python3
"""
PeruCompras AI - Benchmark de Qwen3-8B original.

Evalúa:
- Clasificación: accuracy y macro-F1.
- Extracción: coincidencia exacta por campo y registro.
- JSON válido.
- Latencia de inferencia.

Por defecto utiliza VALIDATION.
TEST requiere --allow-test.
"""

import argparse
import collections
import datetime as dt
import json
import re
import time
from pathlib import Path


def parse_json_object(text):
    # Eliminar bloques de razonamiento, si aparecen.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.S
    ).strip()

    # Eliminar Markdown.
    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.I
        )
        text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (ValueError, TypeError):
        pass

    # Recuperar el primer objeto JSON.
    decoder = json.JSONDecoder()

    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def metrics(rows):
    by_task = collections.defaultdict(list)

    for row in rows:
        by_task[row["task"]].append(row)

    result = {
        "n": len(rows),
        "tasks": {}
    }

    for task, group in by_task.items():
        valid = sum(
            row["parsed"] is not None
            for row in group
        )

        data = {
            "count": len(group),
            "json_valid_rate": round(valid / len(group), 4),
            "mean_latency_seconds": round(
                sum(row["latency_seconds"] for row in group)
                / len(group),
                3
            )
        }

        if task == "classification":
            labels = sorted(set(
                row["expected"].get("category")
                for row in group
            ))

            correct = sum(
                row["parsed"] is not None
                and row["parsed"].get("category")
                == row["expected"].get("category")
                for row in group
            )

            data["accuracy"] = round(
                correct / len(group), 4
            )

            scores = []

            for label in labels:
                tp = sum(
                    row["expected"].get("category") == label
                    and (row["parsed"] or {}).get("category") == label
                    for row in group
                )

                fp = sum(
                    row["expected"].get("category") != label
                    and (row["parsed"] or {}).get("category") == label
                    for row in group
                )

                fn = sum(
                    row["expected"].get("category") == label
                    and (row["parsed"] or {}).get("category") != label
                    for row in group
                )

                denominator = 2 * tp + fp + fn

                scores.append(
                    2 * tp / denominator
                    if denominator else 0
                )

            data["macro_f1"] = (
                round(sum(scores) / len(scores), 4)
                if scores else None
            )

            data["labels_in_gold"] = labels

        elif task == "extraction":
            total = sum(
                len(row["expected"])
                for row in group
            )

            good = sum(
                sum(
                    (row["parsed"] or {}).get(
                        key, "__MISSING__"
                    ) == value
                    for key, value in row["expected"].items()
                )
                for row in group
            )

            exact = sum(
                row["parsed"] == row["expected"]
                for row in group
            )

            data["field_exact_match"] = (
                round(good / total, 4)
                if total else None
            )

            data["record_exact_match"] = round(
                exact / len(group), 4
            )

            data["notice"] = (
                "Mide copia o transformación de atributos "
                "suministrados, no extracción desde PDF."
            )

        result["tasks"][task] = data

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-8B original"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/sft/pilot/val.jsonl")
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/reports/benchmark_base_val"
        )
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B"
    )

    parser.add_argument(
        "--quantization",
        choices=["none", "4bit"],
        default="4bit"
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384
    )

    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=2048
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = todos los ejemplos"
    )

    parser.add_argument(
        "--allow-test",
        action="store_true"
    )

    args = parser.parse_args()

    if (
        args.input.name.lower() == "test.jsonl"
        and not args.allow_test
    ):
        parser.error(
            "TEST reservado. Utiliza --allow-test "
            "solamente con la configuración congelada."
        )

    if args.max_new_tokens < 20:
        parser.error("max-new-tokens inválido")

    if args.max_input_tokens < 128:
        parser.error("max-input-tokens inválido")

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
        parser.error(
            "La cuantización 4bit requiere CUDA operativo."
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True
    )

    items = []

    with args.input.open(
        encoding="utf-8-sig"
    ) as file:

        for line_number, line in enumerate(file, 1):

            if not line.strip():
                continue

            record = json.loads(line)

            instruction = record["instruction"]

            if "Clasifica el objeto contractual" in instruction:
                task = "classification"

            elif "Extrae los atributos verificables" in instruction:
                task = "extraction"

            else:
                raise ValueError(
                    f"Tarea desconocida: línea {line_number}"
                )

            items.append(
                (line_number, record, task)
            )

    if args.limit:
        items = items[:args.limit]

    if not items:
        raise ValueError("Dataset vacío")

    print(
        f"Cargando {args.model} "
        f"en {args.quantization} "
        f"para {len(items)} ejemplos...",
        flush=True
    )

    # Cargar tokenizador.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    model_kwargs = {
        "device_map": "auto",
        "dtype": "auto"
    }

    if args.quantization == "4bit":

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model_kwargs["quantization_config"] = (
            quantization_config
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs
    )

    model.eval()

    device = model.get_input_embeddings().weight.device

    print(
        f"Modelo cargado en {device}",
        flush=True
    )

    results = []

    predictions_path = (
        args.output / "predictions.jsonl"
    )

    with predictions_path.open(
        "w",
        encoding="utf-8"
    ) as destination:

        for index, (line_number, record, task) in enumerate(
            items, 1
        ):

            expected = json.loads(
                record["output"]
            )

            user_prompt = (
                record["instruction"]
                + "\n\nDATOS DE ENTRADA (JSON):\n"
                + record["input"]
                + "\n\nResponde ÚNICAMENTE con "
                  "un objeto JSON válido."
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente de clasificación "
                        "y transformación de datos de "
                        "contrataciones públicas del Perú. "
                        "Usa solo los datos proporcionados. "
                        "Devuelve un único JSON sin "
                        "explicaciones ni Markdown."
                    )
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]

            # CORRECCIÓN PRINCIPAL:
            # Solicitar explícitamente un diccionario
            # de tensores, no un tensor directamente.

            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt"
            )

            # BatchEncoding es un diccionario que
            # contiene input_ids y attention_mask.

            model_inputs = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            input_ids = model_inputs["input_ids"]

            # Ahora shape pertenece a un tensor válido.
            input_length = input_ids.shape[-1]

            if input_length > args.max_input_tokens:
                raise ValueError(
                    f"Entrada demasiado larga en "
                    f"línea {line_number}: "
                    f"{input_length} tokens"
                )

            start = time.perf_counter()

            with torch.inference_mode():

                generated = model.generate(
                    **model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            duration = (
                time.perf_counter() - start
            )

            # Extraer solamente los tokens generados,
            # excluyendo el prompt de entrada.

            generated_ids = generated[0][input_length:]

            completion = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            ).strip()

            parsed = parse_json_object(
                completion
            )

            result = {
                "line": line_number,
                "task": task,
                "expected": expected,
                "parsed": parsed,
                "raw_response": completion,
                "latency_seconds": round(duration, 3),
                "generated_tokens": int(
                    generated.shape[-1] - input_length
                )
            }

            destination.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

            destination.flush()

            results.append(result)

            print(
                f"[{index}/{len(items)}] "
                f"{task} "
                f"json={parsed is not None} "
                f"latencia={duration:.2f}s",
                flush=True
            )

    report = metrics(results)

    report["run"] = {
        "model": args.model,
        "input": str(args.input),
        "quantization": args.quantization,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "n": len(results),
        "date_utc": dt.datetime.now(
            dt.timezone.utc
        ).isoformat(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "enable_thinking": False,
        "do_sample": False,
        "prompt_version": "baseline_v1"
    }

    summary_path = (
        args.output / "benchmark_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n=== RESULTADOS ===")

    print(
        json.dumps(
            report["tasks"],
            ensure_ascii=False,
            indent=2
        )
    )

    print(
        "\nResultados guardados en:",
        args.output
    )


if __name__ == "__main__":
    main()
