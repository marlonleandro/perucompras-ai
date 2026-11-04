
import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftConfig, PeftModel


ADAPTER_PATH = Path(
    "artifacts/perucompras-qwen3-8b-e01"
)

# 1. Verificar adaptador
if not (ADAPTER_PATH / "adapter_config.json").exists():
    raise FileNotFoundError(
        f"No existe adapter_config.json en {ADAPTER_PATH}"
    )

config = PeftConfig.from_pretrained(
    str(ADAPTER_PATH)
)

BASE_MODEL = config.base_model_name_or_path

print("Modelo base:", BASE_MODEL)
print("Adaptador:", ADAPTER_PATH)

# 2. Verificar CUDA
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA no está disponible"
    )

print("GPU:", torch.cuda.get_device_name(0))

# 3. Configuración de cuantización
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 4. Cargar tokenizador
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

# 5. Cargar Qwen3-8B cuantizado
print("Cargando modelo base en 4 bits...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quantization_config,
    device_map={"": 0},
    dtype=torch.float16,
    low_cpu_mem_usage=True,
)

print("Modelo base cargado.")

# 6. Cargar adaptador LoRA
print("Cargando adaptador LoRA...")

model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_PATH),
    is_trainable=False,
)

model.eval()

print("Adaptador cargado correctamente.")

# 7. Crear un caso de prueba
messages = [
    {
        "role": "system",
        "content": (
            "Eres un asistente especializado en "
            "contrataciones públicas del Perú. "
            "Responde únicamente con JSON válido. "
            "La clave debe ser category y las opciones "
            "son Bien, Servicio, Obra y Consultoría de Obra."
        ),
    },
    {
        "role": "user",
        "content": (
            "Clasifica esta contratación: "
            "Servicio de mantenimiento preventivo "
            "y correctivo de servidores informáticos."
        ),
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=True,
    return_tensors="pt",
)

device = model.get_input_embeddings().weight.device

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

# 8. Ejecutar inferencia
print("Generando respuesta...")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

prompt_length = inputs["input_ids"].shape[-1]

response = tokenizer.decode(
    output[0][prompt_length:],
    skip_special_tokens=True,
).strip()

print("\n=== RESPUESTA E01 ===")
print(response)

# 9. Validación básica de JSON
try:
    result = json.loads(response)
    print("\nJSON válido:", isinstance(result, dict))
except json.JSONDecodeError:
    print("\nAdvertencia: respuesta no es JSON válido")

# 10. Mostrar memoria GPU
print(
    "\nVRAM máxima reservada (GB):",
    round(
        torch.cuda.max_memory_reserved() / (1024**3),
        2,
    ),
)
