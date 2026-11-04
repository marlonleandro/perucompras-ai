# Validación del dataset SFT

Estado: **PASS**
Errores: 0 | Advertencias: 0

## Conteos

- train_rows: 159
- train_extraction: 87
- train_classification: 72
- val_rows: 23
- val_classification: 10
- val_extraction: 13
- test_rows: 18
- test_classification: 11
- test_extraction: 7

## Incidencias

## Limitaciones
- La extracción es una tarea de copia de campos, no extracción documental.
- Solo se verifica fuga por procedure_id si se suministran metadatos con split y procedure_id.
- La revisión humana del contenido no se sustituye por estas validaciones.
