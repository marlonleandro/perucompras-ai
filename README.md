# PerúCompras AI — Laboratorio de Fine-Tuning con Qwen3-8B

> **Estado:** ingesta, normalización, auditoría y dataset SFT piloto completados; benchmark del modelo base ejecutado localmente sobre validación. **Fine-tuning aún pendiente.**  
> **Última actualización del README:** 2026-11-04.

## 1. Descripción

**PerúCompras AI** es un proyecto de investigación aplicada orientado a construir y evaluar un modelo de lenguaje especializado en contrataciones públicas peruanas. Utiliza datos oficiales **OECE/CONOSCE**, **Qwen3-8B** como modelo open-weight y **Soup CLI** como herramienta prevista para fine-tuning supervisado (SFT) con LoRA/QLoRA.

El laboratorio persigue dos objetivos complementarios:

1. **Investigación reproducible:** comprobar si el ajuste supervisado mejora las tareas de clasificación del objeto contractual y transformación/extracción estructurada respecto de Qwen3-8B sin ajuste.
2. **Producto aplicable:** sentar las bases de un asistente para proveedores que permita analizar convocatorias, identificar requisitos y evaluar preliminarmente su compatibilidad con el perfil de una empresa, apoyándose en documentos vigentes y evidencia verificable.

**Alcance actual:** clasificación en categorías oficiales y transformación de atributos estructurados de convocatorias a JSON. **No** se ha entrenado todavía un clasificador validado de especialidades TI, ni se han implementado análisis jurídico, evaluación automática de elegibilidad o extracción desde bases/TDR completos.

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
| Framework | PyTorch con CUDA habilitado para inferencia; versión exacta por documentar |
| Librerías de trabajo | `pyarrow`, `transformers`, `accelerate`, `bitsandbytes`, `safetensors`; otras según script |

> **Importante:** 32 GB es la **RAM del sistema**, mientras que 8 GB es la **VRAM de la GPU**. Son recursos distintos. La versión CUDA indicada por `nvidia-smi` no necesariamente coincide con la versión con la que fue compilado PyTorch.

La carga cuantizada de Qwen3-8B en la RTX 5050 se confirmó mediante el mensaje `Modelo cargado en cuda:0`. El benchmark de 23 ejemplos terminó y generó sus archivos, pero sus métricas numéricas todavía no están incorporadas a este README.

### Modelo y entrenamiento previsto

- **Base:** [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B).
- **Fine-tuning previsto:** [Soup CLI](https://github.com/MakazhanAlpamys/Soup), SFT con LoRA/QLoRA y cuantización de 4 bits.
- **Inferencia base:** Transformers + bitsandbytes (4 bits), con modo de razonamiento desactivado en el benchmark cuando la versión del tokenizador lo permite.
- **Publicación futura:** Hugging Face (modelo/adaptador, tarjeta del modelo y dataset redistribuible).
- **Producto futuro:** FastAPI, Angular, PostgreSQL, Qdrant/RAG y reglas deterministas.

El archivo `soup.yaml` raíz contiene una configuración preliminar; **su entrenamiento exitoso aún no está verificado**. `configs/soup.yaml` y otros archivos enumerados más adelante están vacíos.

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
├── .gitignore                         # excluye Excel/Word, datos derivados y .venv
├── README.md
├── pyproject.toml                     # pendiente
├── requirements.txt                   # pendiente
├── soup.yaml                          # configuración SFT/LoRA/4 bits inicial
├── api/                               # futuro backend
├── artifacts/                         # checkpoints/adaptadores; no versionar
├── frontend/                          # futura interfaz
├── notebooks/                         # notebooks por desarrollar
├── configs/
│   ├── experiment.yaml                # pendiente
│   └── soup.yaml                      # pendiente; configuración operativa en raíz
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
│   └── benchmark_qwen3_base.py
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
        └── benchmark_base_val/
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

### 5.6. Benchmark Qwen3-8B original

- `benchmark_qwen3_base.py` ejecuta inferencia de Qwen3-8B **sin fine-tuning**, cuantizado en 4 bits, y calcula métricas separadas por tarea.
- Se corrigió el uso de `BatchEncoding` del tokenizador, accediendo a `model_inputs["input_ids"]` y utilizando `model.generate(**model_inputs)`.
- **Smoke test:** ejecutado con 3 ejemplos y archivos generados en `data/reports/benchmark_smoke/`.
- **Validación:** ejecutada con los 23 ejemplos de `val.jsonl`; salidas disponibles en `data/reports/benchmark_base_val/`.
- **Métricas numéricas de accuracy, macro-F1, exact match, JSON válido y latencia:** pendientes de transcribir desde `benchmark_summary.json` (no incluidas en la información compartida hasta ahora).
- **Test (18 ejemplos):** reservado, sin usar para seleccionar prompts o hiperparámetros.

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

### 6.5. Evaluar modelo base

```powershell
# Smoke test: 3 ejemplos de validación
python .\scripts\benchmark_qwen3_base.py --input data/sft/pilot/val.jsonl --output data/reports/benchmark_smoke --quantization 4bit --limit 3

# Benchmark: los 23 ejemplos de validación
python .\scripts\benchmark_qwen3_base.py --input data/sft/pilot/val.jsonl --output data/reports/benchmark_base_val --quantization 4bit
```

**Importante:** no utilizar `test.jsonl` para iterar sobre prompts o parámetros del modelo. El script exige `--allow-test` al usar ese archivo; reservarlo hasta congelar el protocolo.

## 7. Metodología experimental

Los experimentos propuestos son:

| ID | Modelo | Contexto | Estado |
|---|---|---|---|
| B0 | Qwen3-8B original | Prompt estructurado | Ejecutado sobre Validation (23) |
| B1 | Qwen3-8B original | RAG documental | Pendiente |
| F0 | Qwen3-8B + LoRA/QLoRA | Fine-tuning SFT | Pendiente |
| F1 | Qwen3-8B afinado | RAG documental | Pendiente |

**Métricas:** accuracy, macro-F1, validez de JSON, exact match por campo, exact match por registro y latencia. Las métricas se reportan por tarea: clasificación oficial y transformación estructurada.

**Controles:** mantener prompts y parámetros comparables, registrar versión exacta del modelo y librerías, separar por procedimiento, revisar distribución de etiquetas y reservar Test para evaluación final. El piloto de 18 ejemplos Test es insuficiente para conclusiones robustas; se ampliará el benchmark y se incorporará un conjunto temporal.

## 8. Próximos pasos

- [x] Descarga e inventario de fuentes.
- [x] Inspección de esquema CONOSCE 2024–2026.
- [x] Normalización a procedimientos e ítems.
- [x] Auditoría estructural y tratamiento de monedas.
- [x] Generación de 200 ejemplos SFT piloto.
- [x] Validación automática del dataset.
- [x] Ejecución de benchmark base sobre 23 ejemplos de Validation.
- [ ] Analizar `benchmark_summary.json` y revisar errores de `predictions.jsonl`.
- [ ] Registrar versiones exactas de PyTorch, Transformers, bitsandbytes y Soup.
- [ ] Revisar y ampliar muestras por categoría, en particular TI.
- [ ] Construir un benchmark mayor y temporal, con QA humano.
- [ ] Ejecutar smoke test de fine-tuning con Soup CLI.
- [ ] Entrenar y evaluar Qwen3-8B con QLoRA.
- [ ] Comparar B0/B1/F0/F1 en condiciones equivalentes.
- [ ] Incorporar bases/TDR y recuperación documental RAG.
- [ ] Desarrollar API y frontend de demostración.
- [ ] Preparar dataset card, model card, licencias y publicación en Hugging Face.

## 9. Reproducibilidad, privacidad y Git

Los datos originales, Parquet, JSONL generados y checkpoints pueden ser voluminosos o requerir revisión de licencias. **No publicar indiscriminadamente `data/raw/`, `data/processed/`, `data/sft/`, `artifacts/`, `.venv/` ni secretos.** El [`.gitignore`](./.gitignore) ya está preparado; mantenerlo antes del primer `git add .`, y versionar scripts, esquemas, configuraciones, tests y pequeños fixtures públicos. Guardar manifiestos, checksums, fecha de adquisición y condiciones de uso de las fuentes.

Estado actual de la política de exclusión:

| Excluido | Motivo |
| --- | --- |
| `data/raw/`, `*.xlsx`, `*.xls*`, `*.docx`, `*.doc`, `~$*` | Fuentes originales (~1,4 GB) y bloqueos temporales de Office. |
| `data/processed/`, `data/interim/`, `data/benchmark/`, `*.parquet` | Datos derivados voluminosos y regenerables. |
| `artifacts/`, `outputs/`, `runs/`, `wandb/`, `*.safetensors`, `*.ckpt`, `*.pt`, `*.gguf` | Checkpoints y adaptadores de entrenamiento. |
| `.venv/`, `__pycache__/`, `*.py[cod]` | Entorno virtual y artefactos de Python. |
| `.env`, `.env.*`, `*.pem`, `*.key`, `*.log` | Secretos y registros. |

Se versionan los resultados pequeños que sirven de evidencia reproducible: `data/inventory/` (inventario y perfiles), `data/reports/` (auditoría, validación y benchmarks) y el dataset piloto `data/sft/pilot/`. Para incluir un archivo ignorado de forma deliberada: `git add -f ruta/archivo`.

El modelo especializado no sustituirá evaluaciones legales ni garantizará la elegibilidad de una empresa; sus respuestas sobre requisitos deben incluir evidencia documental verificable y controles deterministas cuando corresponda.

## 10. Referencias

- [Soup CLI — repositorio oficial](https://github.com/MakazhanAlpamys/Soup)
- [Qwen3-8B — Hugging Face](https://huggingface.co/Qwen/Qwen3-8B)
- [OECE — Datos abiertos](https://www.gob.pe/es/14272-acceder-al-portal-de-datos-abiertos-del-oece)
- [CONOSCE — Portal](https://bi.seace.gob.pe/pentaho/api/repos/%3Apublic%3Aportal%3Adatosabiertos.html/content?password=key&userid=public)
- [PyTorch — instalación](https://pytorch.org/get-started/locally/)

---

**Estado actual:** listo para examinar cuantitativamente el benchmark de Qwen3-8B original y preparar el primer experimento controlado de fine-tuning con Soup CLI. Aún no se ha publicado un modelo afinado.
