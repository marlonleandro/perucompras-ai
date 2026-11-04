# PerúCompras AI — Laboratorio de Fine-Tuning con Qwen3-8B

> **Estado:** pipeline de datos, benchmarks B1/B2 y entrenamiento/evaluación E01 completados. **E02 (dataset especializado) pendiente.**  
> **Última actualización del README:** 2026-11-04.

## 1. Descripción

**PerúCompras AI** es un proyecto de investigación aplicada orientado a construir y evaluar un modelo de lenguaje especializado en contrataciones públicas peruanas. Utiliza datos oficiales **OECE/CONOSCE**, **Qwen3-8B** como modelo open-weight y **Soup CLI** para fine-tuning supervisado (SFT) con LoRA/QLoRA. El entrenamiento exploratorio **E01 se completó**, se verificó la carga del adaptador y se comparó contra el modelo original.

El laboratorio persigue dos objetivos complementarios:

1. **Investigación reproducible:** comprobar si el ajuste supervisado mejora las tareas de clasificación del objeto contractual y transformación/extracción estructurada respecto de Qwen3-8B sin ajuste.
2. **Producto aplicable:** sentar las bases de un asistente para proveedores que permita analizar convocatorias, identificar requisitos y evaluar preliminarmente su compatibilidad con el perfil de una empresa, apoyándose en documentos vigentes y evidencia verificable.

**Alcance implementado:** clasificación en cuatro categorías oficiales y transformación de atributos estructurados de convocatorias a JSON. Se entrenó y evaluó un **adaptador LoRA exploratorio (E01)**, sin mejora observable en el conjunto piloto de Validation. **No** se ha implementado todavía un clasificador validado de especialidades TI, análisis jurídico, elegibilidad automática ni extracción de bases/TDR completos.

## 2. Tecnologías y entorno

### Equipo principal de desarrollo

| Componente | Configuración |
|---|---|
| Sistema operativo | Windows de 64 bits (x64) |
| Equipo | Laptop |
| Procesador | Intel Core i5-13450HX, 13.ª generación, 2.40 GHz (frecuencia base mostrada) |
| RAM del sistema | 32 GB instalados (31.7 GB utilizables) |
| GPU dedicada | NVIDIA GeForce RTX 5050 Laptop GPU |
| VRAM dedicada | 8 GB (aprox. 8,151 MiB reportados por `nvidia-smi`) |
| GPU integrada | Intel UHD Graphics |
| Almacenamiento | 954 GB totales; 596.9 GB utilizados según la captura |
| Python | 3.11 |
| NVIDIA | Controlador 610.74; `nvidia-smi` reportó CUDA UMD 13.3 |
| Framework | PyTorch `2.11.0+cu128`; Transformers `5.22.1` en los benchmarks |
| Librerías de trabajo | `pyarrow`, `transformers`, `accelerate`, `bitsandbytes`, `safetensors`; otras según script |

> **Importante:** 32 GB es la **RAM del sistema**, mientras que 8 GB es la **VRAM de la GPU**. Son recursos distintos. La versión CUDA indicada por `nvidia-smi` no necesariamente coincide con la versión con la que fue compilado PyTorch.

La carga cuantizada de Qwen3-8B se confirmó sobre CUDA. El entrenamiento E01 se completó en esta laptop y el adaptador cargó correctamente para inferencia mediante **Transformers + bitsandbytes + PEFT**. En una prueba aislada de inferencia se reportaron **5.79 GB de VRAM máxima reservada**; no representa una medición del máximo usado durante entrenamiento.

### Modelo, herramientas y versiones observadas

- **Base:** [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B); revisión reportada en Soup: `b968826d9c46dd6066d109eabc6255188de91218`.
- **Fine-tuning ejecutado:** [Soup CLI](https://github.com/MakazhanAlpamys/Soup), SFT con adaptador LoRA; configuración de cuantización 4 bits y *layer streaming* para VRAM limitada, conforme a la configuración del experimento.
- **Entrenamiento E01:** `run_20261104_141942_9ce52090`, 36 pasos, 1 época configurada, ~4 minutos.
- **Inferencia:** Transformers + bitsandbytes (NF4 de 4 bits) + PEFT para cargar el adaptador; generación determinista sin modo de razonamiento en el benchmark.
- **Dependencias verificadas en los reportes de benchmark:** PyTorch `2.11.0+cu128`, Transformers `5.22.1`. Versiones exactas de Soup, PEFT, accelerate y bitsandbytes por añadir al manifiesto de entorno.
- **Publicación futura:** Hugging Face, condicionada a evaluación ampliada y revisión de licencias.
- **Producto futuro:** FastAPI, Angular, PostgreSQL, Qdrant/RAG y reglas deterministas.

**Advertencia de hardware:** el entrenamiento con *layer streaming* no implica que la carga de Qwen3-8B en precisión completa para inferencia quepa íntegramente en los 8 GB. `soup chat --model artifacts/perucompras-qwen3-8b-e01` falló por falta de `offload_dir` al distribuir capas hacia CPU; la carga de base cuantizada a 4 bits seguida de `PeftModel.from_pretrained` sí funcionó.

## 3. Fuentes de datos

Los insumos descargados están en `data/raw/` (79 archivos, aproximadamente 1,435 MB según inventario local):

| Familia | Archivos | Cobertura indicada por el nombre |
|---|---:|---|
| CONOSCE — Convocatorias | 3 | 2024, 2025, 2026 |
| CONOSCE — Adjudicaciones | 3 | 2024, 2025, 2026 |
| CONOSCE — Contratos | 3 | 2024, 2025, 2026 |
| CONOSCE — Consorcios | 3 | 2024, 2025, 2026 |
| CONOSCE — Proveedores | 3 | 2024, 2025, 2026 |
| CONOSCE — Órdenes de compra | 34 | Archivos mensuales 2024–2026 disponibles |
| Bases estándar | 30 | Documentos Word `.docx.docx` |
| **Total** | **79** | |

**Fuentes oficiales de referencia:**

- [OECE — Portal de datos abiertos](https://www.gob.pe/es/14272-acceder-al-portal-de-datos-abiertos-del-oece).
- [CONOSCE — Datos abiertos](https://bi.seace.gob.pe/pentaho/api/repos/%3Apublic%3Aportal%3Adatosabiertos.html/content?password=key&userid=public).
- [OECE — Normativa y herramientas de contratación](https://www.gob.pe/institucion/oece/tema/legislacion-y-herramientas-para-las-contrataciones).

**Precauciones:** los años de archivo no garantizan cobertura completa; los registros históricos pueden corresponder a regímenes legales diferentes. El uso y redistribución de los archivos y documentos deberá comprobarse por recurso antes de publicar datasets derivados.

## 4. Estructura del repositorio

```text
perucompras-ai/
├── .gitignore                         # completar/verificar antes de publicar
├── README.md
├── pyproject.toml                     # pendiente
├── requirements.txt                   # pendiente
├── soup.yaml                          # configuración preliminar anterior a E01
├── api/                               # futuro backend
├── artifacts/                         # checkpoints/adaptadores; no versionar
│   └── perucompras-qwen3-8b-e01/      # adaptador LoRA E01 generado por Soup
├── frontend/                          # futura interfaz
├── notebooks/                         # notebooks por desarrollar
├── configs/
│   ├── experiment.yaml                # pendiente
│   ├── soup.yaml                      # archivo anterior/preliminar
│   └── soup_e01.yaml                  # configuración de entrenamiento E01
├── docs/
│   ├── data_sources.md                # pendiente
│   ├── dataset_card.md                # pendiente
│   ├── methodology.md                 # pendiente
│   └── model_card.md                  # pendiente
├── src/perucumpras/                  # nombre actual en disco; revisar antes de crear paquete Python
│   ├── evaluation/
│   ├── inference/
│   ├── ingestion/
│   ├── labeling/
│   ├── preprocessing/
│   └── training/
├── scripts/
│   ├── inventory_sources.py
│   ├── inspect_sources.py
│   ├── normalize_conosce.py
│   ├── common_conosce.py
│   ├── audit_conosce.py
│   ├── build_sft_dataset.py
│   ├── validate_sft_dataset.py
│   ├── benchmark_qwen3_base.py       # baseline v1
│   ├── benchmark_qwen3_base_v2.py    # baseline con contrato JSON explícito
│   ├── benchmark_qwen3_lora_v2.py    # benchmark v2 que carga adaptador PEFT
│   └── test_lora_e01.py              # smoke test del adaptador
├── tests/
│   ├── test_pipeline.py
│   └── test_tools.py
└── data/
    ├── raw/                           # 79 fuentes originales
    ├── interim/                       # vacío
    ├── benchmark/                     # vacío
    ├── inventory/                     # inventario e inspecciones
    ├── processed/
    │   ├── procedures.parquet
    │   ├── procedure_items.parquet
    │   ├── normalization_errors.csv
    │   ├── normalization_summary.json
    │   └── smoke/                     # prueba de normalización CSV
    ├── sft/pilot/
    │   ├── train.jsonl
    │   ├── val.jsonl
    │   ├── test.jsonl
    │   ├── examples_metadata.csv
    │   └── dataset_manifest.json
    └── reports/
        ├── audit/
        ├── sft_validation/
        ├── benchmark_smoke/
        ├── benchmark_base_val/        # baseline v1
        ├── benchmark_base_val_v2/     # baseline v2
        └── benchmark_lora_e01_val/    # evaluación E01 con v2
```

## 5. Pipeline implementado y resultados

### 5.1. Inventario e inspección

- `inventory_sources.py`: clasifica los archivos por familia, período, formato y tamaño; opcionalmente registra hashes.
- `inspect_sources.py`: inspecciona hojas y columnas XLSX, tipos observados, nulos y fechas; ofrece muestreo (`quick`) y recorrido completo (`full`).
- Los archivos `CONOSCE_CONVOCATORIAS2024_0.xlsx`, `...2025_0.xlsx` y `...2026_0.xlsx` presentan **28 columnas** con un esquema coincidente en la muestra inspeccionada.

### 5.2. Normalización

- `normalize_conosce.py`: transforma convocatorias en tablas relacionadas de procedimientos e ítems; conserva referencias de origen y genera informes.
- Se corrigió el reconocimiento de denominaciones de moneda (entre otras: USD, GBP, CAD, CHF y BRL).

| Indicador | Resultado |
|---|---:|
| Filas leídas | 177,945 |
| Filas importadas | 177,945 |
| Procedimientos agrupados | 161,370 |
| Ítems | 177,945 |
| Advertencias tras corrección de monedas | 4 conflictos de moneda a nivel de procedimiento |

**Regla de modelado:** la moneda y el importe del ítem se conservan a nivel de ítem; no se suman importes de monedas distintas. `procedure_id` es una clave del ETL, no un identificador oficial universal verificado para todos los conjuntos de CONOSCE.

### 5.3. Auditoría de integridad

- `audit_conosce.py`: verifica claves, integridad referencial, conteos, calidad de campos, distribuciones y candidatos TI.
- En Windows se corrigió un bloqueo al eliminar `index.sqlite` temporal cerrando explícitamente la conexión antes de limpiar el directorio.

| Control | Incidencias |
|---|---:|
| `procedure_id` duplicado | 0 |
| `procedure_id` ausente | 0 |
| Ítems sin procedimiento | 0 |
| `source_record_id` duplicado | 0 |
| `source_record_id` ausente | 0 |
| Desajuste de cantidad de ítems | 0 |
| Procedimientos con monedas mixtas | 4 |
| Ítems sin `item_code` | 30,091 |

Los 30,091 `item_code` vacíos no se completan artificialmente; se mantiene `source_record_id` como referencia técnica de origen. Se exportaron **300 posibles contrataciones tecnológicas** mediante palabras clave: **no son etiquetas validadas**.

**Procedimientos con monedas mixtas a revisar:** `2024:1063068`, `2025:1172174`, `2026:1189859` y `2026:1260048`.

### 5.4. Dataset supervisado piloto

- `build_sft_dataset.py`: genera ejemplos estilo Alpaca (`instruction`, `input`, `output`) y metadatos de procedencia.
- Separación determinista por `procedure_id` mediante SHA-256 y semilla 42, evitando que el mismo procedimiento aparezca en particiones distintas.
- Las etiquetas oficiales de clasificación provienen de `OBJETOCONTRACTUAL`; **no** se infieren etiquetas TI, cumplimiento legal ni decisiones regulatorias.

| Conjunto | Clasificación | Extracción | Total |
|---|---:|---:|---:|
| Train | 72 | 87 | 159 |
| Validation | 10 | 13 | 23 |
| Test | 11 | 7 | 18 |
| **Total** | **93** | **107** | **200** |

- Candidatos reportados por el generador: **322,740** (no implican igual cantidad de registros independientes).
- Procedimientos seleccionados: **200**.
- Estado de anotación: `AUTO_GENERATED_REQUIRES_QA`; revisión humana pendiente.
- **Límite científico:** la extracción piloto transforma atributos ya proporcionados en la entrada; no demuestra extracción desde PDF ni interpretación de bases o TDR.

### 5.5. Validación SFT

- `validate_sft_dataset.py` revisa JSONL, campos obligatorios, tipos de tarea, duplicados, estructura de respuestas y, cuando se proporciona, metadatos de partición.
- Resultado reportado para el piloto: **200 ejemplos evaluados, 0 errores automáticos**.
- Esta aprobación no equivale a revisión semántica, normativa o humana de las etiquetas.

### 5.6. Benchmark B1 — Qwen3-8B original, prompt genérico

- Script: `scripts/benchmark_qwen3_base.py`; 23 ejemplos de Validation; 4 bits; `max_new_tokens=384`; prompt `baseline_v1`.
- Clasificación: **accuracy 0 %**, macro-F1 **0** (10 ejemplos).
- Extracción: **field exact match 0 %**, **record exact match 0 %** (13 ejemplos); 92.31 % de respuestas JSON válidas en esta tarea.
- **Interpretación:** las respuestas usaban claves diferentes de las esperadas, y el evaluador exigía coincidencia exacta. Los ceros indican **incumplimiento del contrato estricto**, no ausencia absoluta de capacidad semántica.
- Resultado en `data/reports/benchmark_base_val/`.

### 5.7. Benchmark B2 — Qwen3-8B original, contrato explícito

- Script: `scripts/benchmark_qwen3_base_v2.py`; `prompt_version=baseline_v2_schema_contract_1`; contrato de cuatro categorías oficiales y diez campos canónicos; `max_new_tokens=768`; JSON estricto.
- Validation: **23** ejemplos (10 clasificación, 13 extracción), con `input_sha256=df2be79f4ad4139172d3efc971d2e1fdc3fd63449fcaa5988c3fadf64f6099be`.
- Modelo original; PyTorch `2.11.0+cu128`, Transformers `5.22.1`; `do_sample=False`, `enable_thinking=False`, cuantización 4 bits.
- Clasificación: **accuracy 90.00 % (9/10)** y **macro-F1 0.881**.
- Extracción: **field exact match 98.46 % (128/130)**, **record exact match 84.62 % (11/13)**.
- JSON válido y conformidad de esquema: **100 %** en ambas tareas; **0 %** truncamiento detectado.
- Latencia promedio: clasificación **1.046 s**, extracción **26.138 s**; P95: **1.665 s** y **57.458 s**.
- Errores observados: clasificación de un `Bien` como `Obra` (línea 3); normalización no solicitada de `PROVINCIAA` a `PROVINCIA` (línea 21) y de `AU-TORIZADOS` a `AUTORIZADOS` (línea 23).
- Resultado en `data/reports/benchmark_base_val_v2/`.

**No comparar B1 y B2 como si fueran pesos diferentes:** el modelo es el mismo, pero cambiaron el contrato de prompt y el presupuesto de tokens. B2 mide la línea base pertinente para comparar el fine-tuning.

### 5.8. Entrenamiento E01 — Soup CLI, LoRA/QLoRA exploratorio

**Ejecución completada el 04/11/2026** en Windows, RTX 5050 Laptop 8 GB, con `Qwen/Qwen3-8B`.

| Parámetro | Valor verificado o configurado |
|---|---|
| Run ID | `run_20261104_141942_9ce52090` |
| Estado | `completed` |
| Tarea / backend | SFT / Transformers |
| Ejemplos originales de Train | 159; `val_split: 0.1` (aprox. 143 entrenamiento interno) |
| `data.max_length` | **1024** (512 descartaba el objetivo en la fila 11) |
| `data.train_on_responses_only` | `true` |
| `data.train_on_prompt` | `false` |
| Épocas / pasos | 1 época configurada / **36 pasos** |
| Learning rate | `2e-05` |
| Batch / acumulación | 1 / 4 |
| LoRA | `r=8`, `alpha=16` configurado |
| Pérdida reportada por Soup | **0.8130 → 1.1173** |
| Duración reportada | **4 min** |
| Salida | `artifacts/perucompras-qwen3-8b-e01` |

**Incidencias y soluciones:**

1. Con `max_length: 512`, Soup abortó por falta de tokens supervisables en la fila 11 después de truncar; el ajuste a **1024** permitió completar la preparación y el entrenamiento.
2. Transformers mostró una advertencia relativa a `return_assistant_tokens_mask` y `{% generation %}`. La configuración de ejecución reportó entrenamiento sobre respuestas (`train_on_responses_only=true`), pero el log disponible **no demuestra por sí solo** cómo se construyó la máscara en cada ejemplo; conviene inspeccionarla antes de E02.
3. Soup finalizó correctamente y generó el adaptador. `soup chat` falló durante la carga del adaptador por `offload_dir` no configurado al usar CPU offload. El script `test_lora_e01.py` resolvió la inferencia cargando el modelo base **NF4 4-bit** y luego el adaptador con PEFT.
4. **Smoke test de inferencia exitoso:** `{"category":"Servicio"}` ante un caso de mantenimiento de servidores; JSON válido. Reserva máxima de VRAM observada **5.79 GB** durante esta prueba.

**Interpretación del Loss:** Soup mostró un incremento de 0.8130 a 1.1173 entre dos valores resumidos. Sin curva completa ni validación de pérdida, esos puntos no bastan para inferir deterioro o mejora de generalización.

### 5.9. Benchmark F1 — Qwen3-8B + adaptador E01

- Script: `scripts/benchmark_qwen3_lora_v2.py`, que carga el modelo original cuantizado a 4 bits y después el adaptador mediante PEFT.
- Se mantuvieron el mismo dataset Validation, hash, prompt, tamaño máximo de salida y generación determinista de B2.
- Resultado: **23/23 respuestas textuales idénticas a B2**, incluidos los mismos tres ejemplos con discrepancias.

| Métrica | B2: original | F1: E01 LoRA | Diferencia |
|---|---:|---:|---:|
| Accuracy clasificación | 90.00 % | 90.00 % | 0 pp |
| Macro-F1 | 0.881 | 0.881 | 0 |
| Extracción: coincidencia por campo | 98.46 % | 98.46 % | 0 pp |
| Extracción: registros exactos | 84.62 % | 84.62 % | 0 pp |
| JSON válido / conformidad de esquema | 100 % | 100 % | 0 pp |
| Truncamientos detectados | 0 % | 0 % | 0 pp |
| Latencia media clasificación | 1.046 s | 1.181 s | +0.135 s |
| Latencia media extracción | 26.138 s | 25.692 s | −0.446 s |
| P95 clasificación | 1.665 s | 1.894 s | +0.229 s |
| P95 extracción | 57.458 s | 33.944 s | −23.514 s |

Las variaciones de latencia en esta muestra pequeña **no permiten concluir superioridad de rendimiento**. Las salidas idénticas no prueban que los pesos LoRA sean irrelevantes: pueden cambiar las probabilidades internas sin alterar la decodificación determinista.

**Conclusión E01:** ciclo técnico de fine-tuning e inferencia validado en hardware de consumo; **ninguna mejora observable de precisión o de respuestas en Validation (n=23)**. No justifica aún la publicación como modelo especializado superior a Qwen3-8B original.

## 6. Reproducir el trabajo en Windows (PowerShell)

Ejecutar desde la raíz de `perucompras-ai/` y con el entorno virtual activo. Los pasos de transformación requieren sus archivos de entrada existentes; **no es necesario regenerar todo para continuar desde el benchmark**.

### 6.1. Entorno

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pyarrow transformers accelerate bitsandbytes safetensors
```

Instalar PyTorch con soporte CUDA compatible con la RTX 5050 siguiendo el selector oficial: <https://pytorch.org/get-started/locally/>. Antes de reinstalar, verificar el entorno ya existente. Soup CLI se instala por separado cuando se vaya a entrenar:

```powershell
python -m pip install "soup-cli[train]"
```

Comprobaciones:

```powershell
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('Disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No detectada')"
```

### 6.2. Inventario e inspección

```powershell
python .\scripts\inventory_sources.py --raw data/raw --output data/inventory --inspect-xlsx --hash
python .\scripts\inspect_sources.py --raw data/raw --output data/inventory/inspection --family CONVOCATORIAS --mode quick --sample-rows 500
```

### 6.3. Normalizar y auditar

```powershell
python .\scripts\normalize_conosce.py --raw data/raw --output data/processed
python .\scripts\audit_conosce.py --input data/processed --output data/reports/audit
```

### 6.4. Generar y validar SFT piloto

```powershell
python .\scripts\build_sft_dataset.py --input data/processed --output data/sft/pilot --max-examples 200 --seed 42
python .\scripts\validate_sft_dataset.py --input data/sft/pilot --metadata data/sft/pilot/examples_metadata.csv --output data/reports/sft_validation
```

### 6.5. Evaluar baseline B1 y B2

```powershell
# B1: prompt genérico (resultado histórico)
python .\scripts\benchmark_qwen3_base.py --input data/sft/pilot/val.jsonl --output data/reports/benchmark_base_val --quantization 4bit

# B2: prompt y contrato JSON explícitos
python .\scripts\benchmark_qwen3_base_v2.py --input data/sft/pilot/val.jsonl --output data/reports/benchmark_base_val_v2 --quantization 4bit
```

### 6.6. Entrenar E01 con Soup CLI

Archivo `configs/soup_e01.yaml`, correspondiente al experimento (los valores exactos adicionales deben conservarse desde el archivo local ejecutado):

```yaml
base: Qwen/Qwen3-8B
task: sft

data:
  train: ./data/sft/pilot/train.jsonl
  format: alpaca
  val_split: 0.1
  max_length: 1024
  train_on_responses_only: true

training:
  epochs: 1
  lr: 0.00002
  batch_size: 1
  gradient_accumulation_steps: 4
  lora:
    r: 8
    alpha: 16
  quantization: 4bit
  stream_layers: true
  stream_source: auto

output: ./artifacts/perucompras-qwen3-8b-e01
```

```powershell
soup doctor --config configs/soup_e01.yaml
soup train --config configs/soup_e01.yaml --dry-run
soup train --config configs/soup_e01.yaml
soup runs show run_20261104_141942_9ce52090
```

**Nota:** el YAML anterior resume la configuración conocida, pero el registro de ejecución truncado no permite verificar todas las claves/valores. Para reproducibilidad exacta, conservar el YAML realmente ejecutado, la configuración íntegra de `soup runs show` y un `pip freeze`. No modificar retroactivamente el artefacto original.

### 6.7. Cargar y probar el adaptador

```powershell
python .\scripts\test_lora_e01.py
```

Este script carga `Qwen/Qwen3-8B` con `BitsAndBytesConfig(load_in_4bit=True)` y aplica el adaptador guardado mediante `PeftModel.from_pretrained(...)`. Si `soup chat` da un error de `offload_dir`, utilizar la ruta PEFT ya validada en lugar de fusionar el modelo.

### 6.8. Benchmark del modelo afinado E01

```powershell
python .\scripts\benchmark_qwen3_lora_v2.py `
  --input data/sft/pilot/val.jsonl `
  --output data/reports/benchmark_lora_e01_val `
  --model Qwen/Qwen3-8B `
  --adapter artifacts/perucompras-qwen3-8b-e01 `
  --quantization 4bit `
  --max-new-tokens 768
```

Para evaluar correctamente, utilizar **el mismo** prompt, dataset, límites de tokens, cuantización y opciones de decodificación del baseline B2. Evitar usar Test para iteraciones de ajuste.

## 7. Metodología experimental y conclusiones

| Experimento | Pesos | Evaluación | Estado |
|---|---|---|---|
| B1 | Qwen3-8B original | Prompt genérico; 23 Validation | Completado; 0 % según coincidencia estricta |
| B2 | Qwen3-8B original | Contrato explícito v2; 23 Validation | Completado; **90 % clasificación, 98.46 % por campo** |
| E01 / F1 | Qwen3-8B + LoRA Soup | Mismo contrato v2; 23 Validation | Completado; **sin mejora observable** |
| E02 | Qwen3-8B + entrenamiento especializado | Benchmark avanzado por construir | Pendiente |
| RAG documental | Modelo con recuperación de evidencia | Requisitos normativos y bases/TDR | Pendiente |

**Hallazgo B1→B2:** establecer el contrato de respuesta influyó decisivamente en el resultado medido. B1/B2 no constituyen una comparación de pesos entrenados porque el prompt y el presupuesto de tokens difieren.

**Hallazgo B2→E01:** el primer fine-tuning fue técnicamente exitoso, pero no cambió ninguna de las 23 salidas. La muestra (10 clasificación, 13 extracción) es pequeña, y la tarea de extracción principalmente copia/renombra información disponible, por lo que no mide comprensión profunda del dominio.

**Riesgos metodológicos:** etiquetas automáticas pendientes de QA; pocos ejemplos por clase; precisión alta de partida; no hay evaluación independiente de conocimiento normativo; un único entrenamiento no permite estimar variabilidad entre semillas. Las métricas de Test (18 casos) permanecen reservadas. No atribuir al fine-tuning mejoras de contenido sin controlar prompt, particiones, decodificación y revisión de fuentes.

## 8. Próximo experimento E02 — Dataset especializado (pendiente)

**Objetivo:** estudiar si el SFT aporta capacidades de interpretación especializada sobre contratación pública peruana que no se resuelvan suficientemente mediante instrucciones, transformación determinista o RAG.

**Propuesta de trabajo, todavía no ejecutada:**

1. Revisar las **30 bases estándar** y documentar procedencia, edición, vigencia, marco normativo y permisos de reutilización. Integrar fuentes normativas oficiales vigentes cuando corresponda.
2. Diseñar tareas con dificultad real: identificación de requisitos de participación, plazos y criterios, extracción con evidencia, clasificación ambigua y detección de información insuficiente. Separar hechos documentales de inferencias.
3. Construir `build_sft_dataset_v2.py` con trazabilidad por documento/sección y estado de revisión humana; no utilizar reglas normativas inventadas como etiquetas.
4. Crear un nuevo benchmark independiente, estratificado por tipo de documento, categoría y dificultad; separar train/validation/test por documento y, cuando sea viable, por período.
5. Evaluar previamente el **modelo original con prompt fuerte**, y si procede también **RAG sin fine-tuning**, para establecer baselines competitivos.
6. Afinar únicamente con ejemplos revisados y evaluar contra los mismos baselines. Registrar pérdidas por paso, máscaras de supervisión, hiperparámetros, semillas, tiempo, VRAM y análisis cualitativo de errores.

**Criterios de éxito:** aumento medible de exactitud semántica y fidelidad a evidencias en muestras de evaluación no vistas, sin deterioro relevante en validez JSON, tasa de respuestas sustentadas ni errores graves. Reportar incertidumbre por tamaño de muestra; no concluir generalización con 23 ejemplos.

**Próximos pasos operativos:**

- [x] Ingesta, normalización y auditoría de CONOSCE.
- [x] Dataset piloto de 200 ejemplos y particiones por procedimiento.
- [x] Benchmark B1 y contrato v2 B2.
- [x] Fine-tuning Soup E01 (36 pasos, 4 min) y prueba de inferencia PEFT.
- [x] Benchmark E01 comparado con B2: 23/23 respuestas idénticas.
- [ ] Respaldar YAML efectivo, `pip freeze`, hash de adaptador y logs completos E01.
- [ ] Revisar las máscaras supervisadas y confirmar el funcionamiento de `train_on_responses_only`.
- [ ] Diseñar esquemas y rúbricas de anotación especializada E02.
- [ ] Inspeccionar bases estándar y versiones normativas aplicables.
- [ ] Generar y revisar manualmente dataset y benchmark E02.
- [ ] Ejecutar baseline fuerte y nuevo SFT especializado.
- [ ] Añadir RAG, API y demo; completar model card y dataset card.
- [ ] Revisar licencias antes de publicar pesos/datos en Hugging Face.

## 9. Reproducibilidad, privacidad y Git

Los datos originales, Parquet, JSONL generados y checkpoints pueden ser voluminosos o requerir revisión de licencias. **No publicar indiscriminadamente `data/raw/`, `data/processed/`, `data/sft/`, `artifacts/`, `.venv/` ni secretos.** Preparar un `.gitignore` antes del primer `git add .`, y versionar scripts, esquemas, configuraciones, tests y pequeños fixtures públicos. Guardar manifiestos, checksums, fecha de adquisición y condiciones de uso de las fuentes.

El modelo especializado no sustituirá evaluaciones legales ni garantizará la elegibilidad de una empresa; sus respuestas sobre requisitos deben incluir evidencia documental verificable y controles deterministas cuando corresponda.

## 10. Referencias

- [Soup CLI — repositorio oficial](https://github.com/MakazhanAlpamys/Soup)
- [Qwen3-8B — Hugging Face](https://huggingface.co/Qwen/Qwen3-8B)
- [OECE — Datos abiertos](https://www.gob.pe/es/14272-acceder-al-portal-de-datos-abiertos-del-oece)
- [CONOSCE — Portal](https://bi.seace.gob.pe/pentaho/api/repos/%3Apublic%3Aportal%3Adatosabiertos.html/content?password=key&userid=public)
- [PyTorch — instalación](https://pytorch.org/get-started/locally/)

---

**Estado actual:** E01 entrenado y evaluado; se comprobó la viabilidad técnica de LoRA en una RTX 5050 de 8 GB, **sin mejora observable sobre B2**. E02 está planificado, no ejecutado. No se ha publicado un modelo afinado.
