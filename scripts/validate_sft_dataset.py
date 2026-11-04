#!/usr/bin/env python3
"""Valida datasets Alpaca de PeruCompras AI sin dependencias externas.
Salida: report.json, report.md, issues.csv. Exit 0 si no hay errores; 2 si hay errores.
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

SPLITS = ('train', 'val', 'test')
CLASS_LABELS = {'Bien', 'Servicio', 'Obra', 'Consultoría de Obra'}
CLASS_INPUT = {'descripcion_del_item', 'objeto_del_proceso'}
EXTRACT_MAP = {
 'category': 'objeto_contractual', 'description': 'descripcion_del_item',
 'entity_name': 'entidad', 'item_currency': 'moneda_item',
 'item_estimated_value': 'monto_referencial_item',
 'procedure_currency': 'moneda_proceso',
 'procedure_estimated_value': 'monto_referencial_proceso',
 'procurement_type': 'tipo_de_seleccion', 'publication_date': 'fecha_convocatoria',
 'title': 'objeto_del_proceso',
}

def issue(issues, severity, split, line, code, detail):
    issues.append(dict(severity=severity, split=split, line=line, code=code, detail=str(detail)[:600]))

def load_file(path, split, issues):
    records=[]
    if not path.is_file():
        issue(issues,'ERROR',split,0,'missing_file',str(path))
        return records
    with path.open(encoding='utf-8-sig') as f:
        for line_no,line in enumerate(f,1):
            if not line.strip():
                issue(issues,'WARNING',split,line_no,'blank_line','Línea vacía')
                continue
            try: rec=json.loads(line)
            except json.JSONDecodeError as e:
                issue(issues,'ERROR',split,line_no,'invalid_jsonl',e)
                continue
            if not isinstance(rec,dict):
                issue(issues,'ERROR',split,line_no,'not_object','Registro debe ser JSON object')
                continue
            records.append((line_no,rec))
    if not records:
        issue(issues,'ERROR',split,0,'empty_split','Sin registros válidos')
    return records

def identify(rec):
    s=rec.get('instruction','')
    if isinstance(s,str) and 'Clasifica el objeto contractual' in s: return 'classification'
    if isinstance(s,str) and 'Extrae los atributos verificables' in s: return 'extraction'
    return 'unknown'

def validate(folder, metadata=None):
    issues=[]; counts=collections.Counter(); class_dist=collections.defaultdict(collections.Counter)
    seen_examples={}; seen_inputs={}; normalized_inputs={}; canonical_output={}
    data={}; split_fingerprints=collections.defaultdict(set)
    for split in SPLITS:
        data[split]=load_file(folder/f'{split}.jsonl',split,issues)
        for n,rec in data[split]:
            counts[f'{split}_rows']+=1
            if set(rec)!= {'instruction','input','output'}:
                issue(issues,'WARNING',split,n,'alpaca_keys',f'Claves presentes: {sorted(rec)}')
            if any(not isinstance(rec.get(k),str) or not rec[k].strip() for k in ('instruction','input','output')):
                issue(issues,'ERROR',split,n,'required_strings','instruction/input/output deben ser strings no vacíos');continue
            try:
                inp=json.loads(rec['input']);out=json.loads(rec['output'])
                if not isinstance(inp,dict) or not isinstance(out,dict): raise ValueError('input/output deben ser objetos JSON')
            except (ValueError,TypeError) as e:
                issue(issues,'ERROR',split,n,'invalid_embedded_json',e);continue
            task=identify(rec);counts[f'{split}_{task}']+=1
            if task=='unknown':
                issue(issues,'ERROR',split,n,'unknown_task','Instrucción no reconocida');continue
            key=json.dumps([rec['instruction'],inp],ensure_ascii=False,sort_keys=True)
            fingerprint=hashlib.sha256(key.encode()).hexdigest();split_fingerprints[split].add(fingerprint)
            if fingerprint in seen_inputs:
                oldsplit,oldline=seen_inputs[fingerprint]
                issue(issues,'ERROR' if oldsplit!=split else 'WARNING',split,n,'duplicate_input',f'Ya presente en {oldsplit}:{oldline}')
            else:seen_inputs[fingerprint]=(split,n)
            if fingerprint in canonical_output and canonical_output[fingerprint]!=out:
                issue(issues,'ERROR',split,n,'contradictory_label','Misma entrada con salida distinta')
            canonical_output[fingerprint]=out
            if task=='classification':
                if set(inp)!=CLASS_INPUT:
                    issue(issues,'WARNING',split,n,'classification_input_schema',str(sorted(inp)))
                if set(out)!={'category'} or out.get('category') not in CLASS_LABELS:
                    issue(issues,'ERROR',split,n,'invalid_class_label',str(out))
                if any(k in inp for k in ['category','objeto_contractual']):
                    issue(issues,'ERROR',split,n,'classification_label_leak','La categoría se encuentra explícita en entrada')
                class_dist[split][str(out.get('category'))]+=1
            else:
                if set(out)!=set(EXTRACT_MAP):
                    issue(issues,'ERROR',split,n,'extraction_output_schema',f'Esperadas {sorted(EXTRACT_MAP)}; recibidas {sorted(out)}')
                if any(k not in inp for k in EXTRACT_MAP.values()):
                    issue(issues,'ERROR',split,n,'extraction_input_schema','Faltan campos de entrada para transformación')
                for out_key,in_key in EXTRACT_MAP.items():
                    if in_key in inp and out_key in out and out[out_key]!=inp[in_key]:
                        issue(issues,'ERROR',split,n,'copy_mismatch',f'{out_key} != {in_key}')
                # Notice deliberate copy task: do not count as a defect.
                if 'publication_date' in out and out['publication_date'] is not None:
                    try:dt.date.fromisoformat(str(out['publication_date']))
                    except ValueError:issue(issues,'WARNING',split,n,'invalid_iso_date',str(out['publication_date']))
    for a,b in [('train','val'),('train','test'),('val','test')]:
        common=split_fingerprints[a] & split_fingerprints[b]
        if common:issue(issues,'ERROR',f'{a}/{b}',0,'cross_split_same_input',len(common))
    meta_summary={}
    if metadata:
        if metadata.is_file():
            with metadata.open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            meta_summary={'rows':len(rows),'columns':list(rows[0]) if rows else []}
            possible_split=next((x for x in ('split','dataset_split') if x in (rows[0] if rows else {})),None)
            possible_proc=next((x for x in ('procedure_id','procedure_key') if x in (rows[0] if rows else {})),None)
            if possible_split and possible_proc:
                proc_split={}
                for r in rows:
                    proc=r[possible_proc]; split=r[possible_split]
                    if not proc:issue(issues,'ERROR','metadata',0,'missing_procedure','procedure_id vacío');continue
                    if proc in proc_split and proc_split[proc]!=split:
                        issue(issues,'ERROR','metadata',0,'procedure_leak',f'{proc}: {proc_split[proc]} / {split}')
                    proc_split[proc]=split
                meta_summary['unique_procedures']=len(proc_split)
            else: issue(issues,'WARNING','metadata',0,'metadata_columns','No se pudo comprobar particionado por procedimiento: faltan split/procedure_id')
            if len(rows)!=sum(len(x) for x in data.values()):
                issue(issues,'ERROR','metadata',0,'metadata_count','Número de filas no coincide')
        else:issue(issues,'ERROR','metadata',0,'missing_metadata',str(metadata))
    summary={
        'status':'FAIL' if any(x['severity']=='ERROR' for x in issues) else 'PASS_WITH_WARNINGS' if issues else 'PASS',
        'counts':dict(counts),'class_distribution':{k:dict(v) for k,v in class_dist.items()},
        'issues_by_code':dict(collections.Counter(x['code'] for x in issues)),
        'errors':sum(x['severity']=='ERROR' for x in issues),
        'warnings':sum(x['severity']=='WARNING' for x in issues),
        'metadata':meta_summary,
        'limitations':['La extracción es una tarea de copia de campos, no extracción documental.',
                       'Solo se verifica fuga por procedure_id si se suministran metadatos con split y procedure_id.',
                       'La revisión humana del contenido no se sustituye por estas validaciones.']}
    return summary,issues

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('data/sft/pilot'))
    p.add_argument('--output',type=Path,default=Path('data/reports/sft_validation'))
    p.add_argument('--metadata',type=Path,default=None)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    metadata=args.metadata or (args.input/'examples_metadata.csv' if (args.input/'examples_metadata.csv').exists() else None)
    summary,issues=validate(args.input,metadata)
    (args.output/'validation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (args.output/'validation_issues.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['severity','split','line','code','detail']);w.writeheader();w.writerows(issues)
    md=['# Validación del dataset SFT','',f"Estado: **{summary['status']}**",f"Errores: {summary['errors']} | Advertencias: {summary['warnings']}",'','## Conteos','']
    md.extend(f'- {k}: {v}' for k,v in summary['counts'].items())
    md+=['','## Incidencias'];md.extend(f'- {k}: {v}' for k,v in summary['issues_by_code'].items())
    md+=['','## Limitaciones'];md.extend(f'- {x}' for x in summary['limitations'])
    (args.output/'validation_report.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'errors':summary['errors'],'warnings':summary['warnings'],'output':str(args.output)},ensure_ascii=False))
    raise SystemExit(2 if summary['errors'] else 0)
if __name__=='__main__':main()
