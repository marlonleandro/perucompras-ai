"""Utilidades compartidas para auditar Parquet CONOSCE sin cargarlo completo."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def require_arrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit('Falta pyarrow. Ejecuta: python -m pip install pyarrow') from exc
    return pq


def iter_parquet(path: Path, columns=None, batch_size=8192):
    pq = require_arrow()
    pf = pq.ParquetFile(path)
    present = set(pf.schema_arrow.names)
    if columns:
        missing = set(columns) - present
        if missing:
            raise ValueError(f'Columnas ausentes en {path.name}: {sorted(missing)}')
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            yield row


def normalized(value):
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def stable_hash(*parts: object):
    return hashlib.sha256('|'.join(str(x or '') for x in parts).encode('utf-8')).hexdigest()


def choose_split(procedure_id: str, seed=42, train=0.8, val=0.1):
    if train <= 0 or val < 0 or train + val >= 1:
        raise ValueError('Proporciones invalidas: train > 0, val >= 0 y train + val < 1')
    fraction = int(stable_hash(seed, procedure_id)[:15], 16) / float(16**15)
    return 'train' if fraction < train else ('val' if fraction < train + val else 'test')


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
