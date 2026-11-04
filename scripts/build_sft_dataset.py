#!/usr/bin/env python3
"""Construye SFT Alpaca a partir de convocatorias CONOSCE normalizadas.

Solo genera etiquetas derivadas de campos oficiales; no infiere cumplimiento legal
ni especialidades TI. Split estable por procedure_id, metadata en archivo separado.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import heapq
import json
from pathlib import Path
from common_conosce import iter_parquet, normalized, stable_hash, choose_split, write_json, require_arrow

INSTRUCTION_CLASSIFY='Clasifica el objeto contractual de la contratacion peruana usando exclusivamente el texto proporcionado. Devuelve JSON con la categoria oficial.'
INSTRUCTION_EXTRACT='Extrae los atributos verificables de esta convocatoria peruana. Utiliza exclusivamente el registro proporcionado; no inventes datos. Devuelve JSON.'


def make_examples(p, item):
    """Devuelve pares (tarea, ejemplo Alpaca), 100% sustentados por fuentes."""
    title=normalized(p.get('title'))
    desc=normalized(item.get('description')) if item else None
    cat=normalized(p.get('category'))
    pid=normalized(p.get('procedure_id'))
    examples=[]
    # Clasificacion: etiqueta oficial NO aparece en input (evita fuga directa).
    if cat and (title or desc):
        inp=json.dumps({'objeto_del_proceso':title,'descripcion_del_item':desc},ensure_ascii=False,sort_keys=True)
        output=json.dumps({'category':cat},ensure_ascii=False,sort_keys=True)
        examples.append(('classification',{'instruction':INSTRUCTION_CLASSIFY,'input':inp,'output':output}))
    # Extraccion: se suministra un registro parcial como evidencia; no se pide al modelo
    # inferir datos no suministrados. Campos ausentes son null de forma explicita.
    raw={'entidad':normalized(p.get('entity_name')),'objeto_del_proceso':title,
         'descripcion_del_item':desc,'tipo_de_seleccion':normalized(p.get('procurement_type')),
         'objeto_contractual':cat,'fecha_convocatoria':normalized(p.get('publication_date')),
         'monto_referencial_proceso':normalized(p.get('estimated_value')),
         'moneda_proceso':normalized(p.get('currency')),
         'monto_referencial_item':normalized(item.get('estimated_value')) if item else None,
         'moneda_item':normalized(item.get('currency')) if item else None}
    if title or desc:
        output={'entity_name':raw['entidad'],'title':title,'description':desc,
          'procurement_type':raw['tipo_de_seleccion'],'category':cat,
          'publication_date':raw['fecha_convocatoria'],
          'procedure_estimated_value':raw['monto_referencial_proceso'],
          'procedure_currency':raw['moneda_proceso'],
          'item_estimated_value':raw['monto_referencial_item'],
          'item_currency':raw['moneda_item']}
        examples.append(('extraction',{'instruction':INSTRUCTION_EXTRACT,
          'input':json.dumps(raw,ensure_ascii=False,sort_keys=True),
          'output':json.dumps(output,ensure_ascii=False,sort_keys=True)}))
    return examples


def scan_candidates(proc_path, item_path, limit, seed, exclude_years):
    """Seleccion global de menor hash, independiente del orden de registros."""
    pq=require_arrow()
    proc_cols=['procedure_id','source_year','entity_name','title','category','procurement_type','publication_date','estimated_value','currency','source_file','source_sheet','source_excel_row']
    item_cols=['procedure_id','source_record_id','description','estimated_value','currency']
    for name, cols, path in [('procedures',proc_cols,proc_path),('items',item_cols,item_path)]:
        missing=set(cols)-set(pq.ParquetFile(path).schema_arrow.names)
        if missing:raise ValueError(f'{name}: faltan campos {sorted(missing)}')
    # Mapeo compacto de un solo item por procedimiento, no cargar el texto completo.
    # Para una muestra de ~178k filas la memoria es moderada; procesos mayores
    # deberian reemplazar esto con SQLite o Parquet ordenado.
    first_item={}
    for row in iter_parquet(item_path,columns=item_cols):
        pid=normalized(row.get('procedure_id'))
        if pid and pid not in first_item:first_item[pid]=row
    heap=[]
    considered=0
    for p in iter_parquet(proc_path,columns=proc_cols):
        pid=normalized(p.get('procedure_id'))
        if not pid or str(p.get('source_year')) in exclude_years:continue
        item=first_item.get(pid)
        for task,example in make_examples(p,item):
            considered+=1
            eid=stable_hash('perucompras_sft_v1',pid,task)
            score=int(stable_hash(seed,eid)[:16],16)
            metadata={'example_id':eid,'procedure_id':pid,'task':task,
                'source_year':p.get('source_year'),'source_file':p.get('source_file'),
                'source_sheet':p.get('source_sheet'),'source_excel_row':p.get('source_excel_row'),
                'source_record_id':item.get('source_record_id') if item else None,
                'label_origin':'official_field_or_direct_projection',
                'annotation_status':'AUTO_GENERATED_REQUIRES_QA'}
            entry=(-score,eid,example,metadata)
            if len(heap)<limit:heapq.heappush(heap,entry)
            elif entry[0]>heap[0][0]:heapq.heapreplace(heap,entry)
    return sorted(heap,key=lambda t:t[1]), considered


def main(argv=None):
    parser=argparse.ArgumentParser(description='Genera SFT Alpaca reproducible desde Parquet CONOSCE')
    parser.add_argument('--input',type=Path,default=Path('data/processed'))
    parser.add_argument('--output',type=Path,default=Path('data/sft/pilot'))
    parser.add_argument('--max-examples',type=int,default=200)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--train-ratio',type=float,default=0.8)
    parser.add_argument('--val-ratio',type=float,default=0.1)
    parser.add_argument('--exclude-years',nargs='*',default=[],help='Anos reservados para prueba temporal futura')
    args=parser.parse_args(argv)
    if args.max_examples<1:parser.error('--max-examples debe ser mayor a cero')
    try:choose_split('test',args.seed,args.train_ratio,args.val_ratio)
    except ValueError as e:parser.error(str(e))
    proc=args.input/'procedures.parquet';items=args.input/'procedure_items.parquet'
    for f in (proc,items):
        if not f.is_file():parser.error(f'No existe: {f}')
    selected,candidate_count=scan_candidates(proc,items,args.max_examples,args.seed,set(args.exclude_years))
    args.output.mkdir(parents=True,exist_ok=True)
    counts=Counter();taskcounts=Counter(); procedure_splits={}
    paths={s:(args.output/(s+'.jsonl')).open('w',encoding='utf-8') for s in ('train','val','test')}
    with (args.output/'examples_metadata.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['example_id','split','procedure_id','task','source_year','source_file','source_sheet','source_excel_row','source_record_id','label_origin','annotation_status']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        try:
            for _,eid,ex,meta in selected:
                pid=meta['procedure_id']
                split=choose_split(pid,args.seed,args.train_ratio,args.val_ratio)
                if pid in procedure_splits and procedure_splits[pid]!=split:
                    raise AssertionError('Fuga de procedimiento entre splits')
                procedure_splits[pid]=split
                paths[split].write(json.dumps(ex,ensure_ascii=False)+'\n')
                writer.writerow({'split':split,**meta});counts[split]+=1;taskcounts[meta['task']]+=1
        finally:
            for stream2 in paths.values():stream2.close()
    manifest={'schema_version':'perucompras-sft-v1','seed':args.seed,'max_examples':args.max_examples,
      'eligible_candidates':candidate_count,'selected_examples':sum(counts.values()),
      'split_counts':dict(counts),'task_counts':dict(taskcounts),'unique_procedures_selected':len(procedure_splits),
      'split_strategy':'stable sha256 of procedure_id; group level, not row level',
      'label_policy':'Category copied from official OBJETOCONTRACTUAL; extraction copies provided fields; NO inferred IT labels, regulatory decisions, or legal conclusions.',
      'qa_required':True,'notes':['El test no debe usarse para seleccionar ejemplos ni afinar prompts.',
       'Las mismas fuentes pueden revelar la respuesta en el input de extraccion: evaluar extraccion vs transformacion por separado.',
       'Muestra piloto; distribucion por clase no estratificada.','Un procedimiento puede producir mas de un ejemplo; comparte siempre split.'],
      'source_files':[str(proc),str(items)],'excluded_years':args.exclude_years}
    write_json(args.output/'dataset_manifest.json',manifest)
    print('Dataset generado:',dict(counts),'tareas:',dict(taskcounts),'salida:',args.output)

if __name__=='__main__':main()
