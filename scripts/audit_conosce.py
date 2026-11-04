#!/usr/bin/env python3
"""Auditoría integral y no destructiva de los dos Parquet de normalize_conosce.py.

Procesa por lotes, almacena claves en SQLite temporal y crea CSV/JSON/Markdown.
"""
from __future__ import annotations
import argparse
from contextlib import closing
from collections import Counter, defaultdict
import csv
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
import sqlite3
import tempfile
from common_conosce import iter_parquet, normalized, require_arrow, write_json

TECH_RX = re.compile(r'(?i)\b(?:software|sistema(?:s)? de informaci[oó]n|desarrollo de (?:sistemas|software|aplicaciones)|aplicaci[oó]n(?:es)? (?:web|m[oó]vil)|infraestructura (?:ti|tecnol[oó]gica)|servidor(?:es)?|licencia(?:s|miento)? (?:de )?(?:software|inform[aá]tica)|ciberseguridad|base(?:s)? de datos|computadora(?:s)?|equipo(?:s)? de c[oó]mputo|soporte (?:t[eé]cnico|inform[aá]tico)|redes inform[aá]ticas|cloud computing)\b')

def clean_date(s):
    v = normalized(s)
    if not v: return None
    try: return date.fromisoformat(v[:10])
    except ValueError: return None

def valid_amount(s):
    v = normalized(s)
    if v is None: return None
    try:
        d = Decimal(v)
        return d if d.is_finite() else None
    except InvalidOperation: return None

def csv_out(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader();w.writerows(rows)

def main(argv=None):
    ap=argparse.ArgumentParser(description='Audita integridad, calidad y candidatos SFT de CONOSCE')
    ap.add_argument('--input',type=Path,default=Path('data/processed'))
    ap.add_argument('--output',type=Path,default=Path('data/reports/audit'))
    ap.add_argument('--max-review',type=int,default=300,help='Maximo de ejemplos en CSV de revision')
    args=ap.parse_args(argv)
    procpath=args.input/'procedures.parquet'; itempath=args.input/'procedure_items.parquet'
    for f in (procpath,itempath):
        if not f.is_file(): ap.error(f'Archivo no encontrado: {f}')
    args.output.mkdir(parents=True,exist_ok=True)
    pq=require_arrow()
    schemas={x.name:pq.ParquetFile(x).schema_arrow.names for x in (procpath,itempath)}
    needed_p={'procedure_id','source_year','entity_name','title','category','procurement_type','publication_date','estimated_value','currency','item_count'}
    needed_i={'procedure_id','source_record_id','source_year','description','item_number','item_code','estimated_value','currency'}
    for path,required in ((procpath,needed_p),(itempath,needed_i)):
        diff=required-set(schemas[path.name])
        if diff:ap.error(f'Faltan columnas en {path.name}: {sorted(diff)}')
    pcounts=Counter(); icounts=Counter(); warnings=Counter(); cats=Counter(); types=Counter(); years=Counter()
    money=Counter(); annual=Counter(); tech_candidates=[]; problems=[]
    nulls={'procedures':Counter(),'procedure_items':Counter()}
    counts={'procedures':0,'procedure_items':0}
    pfields=['procedure_id','source_year','entity_name','title','category','procurement_type','publication_date','estimated_value','currency','item_count']
    ifields=['procedure_id','source_record_id','source_year','description','item_number','item_code','estimated_value','currency','source_file','source_excel_row']
    available_i=[f for f in ifields if f in schemas[itempath.name]]
    with tempfile.TemporaryDirectory(prefix='pc_audit_') as folder:
        with closing(sqlite3.connect(str(Path(folder)/'index.sqlite'))) as conn:
            conn.execute('CREATE TABLE p (pid TEXT PRIMARY KEY, reported_items INTEGER, currency TEXT, source_year TEXT)')
            conn.execute('CREATE TABLE i (pid TEXT, source_id TEXT, currency TEXT)')
            conn.execute('CREATE INDEX idx_i_pid ON i(pid)')
            conn.execute('CREATE INDEX idx_i_source ON i(source_id)')
            for r in iter_parquet(procpath,columns=pfields):
                counts['procedures']+=1
                for f in pfields:
                    if normalized(r.get(f)) is None:nulls['procedures'][f]+=1
                pid=normalized(r['procedure_id']); pcounts['missing_id']+=not bool(pid)
                if not pid:
                    if len(problems)<args.max_review:problems.append({'kind':'procedure_without_id','procedure_id':'','source_id':'','detail':''})
                    continue
                try:
                    conn.execute('INSERT INTO p VALUES (?,?,?,?)',(pid,int(r['item_count']) if str(r['item_count'] or '').isdigit() else None,normalized(r['currency']),str(r['source_year'])))
                except sqlite3.IntegrityError:
                    pcounts['duplicate_procedure_id']+=1
                    if len(problems)<args.max_review:problems.append({'kind':'duplicate_procedure_id','procedure_id':pid,'source_id':'','detail':''})
                cats[normalized(r['category']) or '<NULO>']+=1
                types[normalized(r['procurement_type']) or '<NULO>']+=1
                years[str(r['source_year'])]+=1
                if normalized(r.get('publication_date')) and not clean_date(r['publication_date']):warnings['invalid_publication_date']+=1
                amount=valid_amount(r['estimated_value'])
                if normalized(r['estimated_value']) and amount is None:warnings['invalid_procedure_amount']+=1
                if amount is not None and amount<0:warnings['negative_procedure_amount']+=1
                if (normalized(r.get('title')) and TECH_RX.search(r['title'])) and len(tech_candidates)<args.max_review:
                    tech_candidates.append({'procedure_id':pid,'source_year':r['source_year'],'title':r['title'],'category':r['category'],'reason':'keyword_title','label_status':'REQUIERE_REVISION'})
            conn.commit()
            for r in iter_parquet(itempath,columns=available_i):
                counts['procedure_items']+=1
                for f in needed_i:
                    if normalized(r.get(f)) is None:nulls['procedure_items'][f]+=1
                pid=normalized(r.get('procedure_id')); sid=normalized(r.get('source_record_id'))
                if not pid:icounts['missing_procedure_id']+=1
                if not sid:icounts['missing_source_record_id']+=1
                conn.execute('INSERT INTO i VALUES (?,?,?)',(pid,sid,normalized(r.get('currency'))))
                money[normalized(r.get('currency')) or '<NULO>']+=1
                annual[str(r.get('source_year'))]+=1
                amount=valid_amount(r.get('estimated_value'))
                if normalized(r.get('estimated_value')) and amount is None:warnings['invalid_item_amount']+=1
                if amount is not None and amount<0:warnings['negative_item_amount']+=1
                if normalized(r.get('description')) is None:warnings['missing_item_description']+=1
                if r.get('description') and TECH_RX.search(str(r['description'])) and len(tech_candidates)<args.max_review:
                    tech_candidates.append({'procedure_id':pid,'source_year':r.get('source_year'),'title':str(r['description'])[:500],'category':'','reason':'keyword_item','label_status':'REQUIERE_REVISION'})
            conn.commit()
            orphans=conn.execute('SELECT COUNT(*) FROM i LEFT JOIN p ON i.pid=p.pid WHERE p.pid IS NULL').fetchone()[0]
            unique_ids=conn.execute('SELECT COUNT(*) FROM (SELECT source_id FROM i WHERE source_id IS NOT NULL GROUP BY source_id HAVING COUNT(*)>1)').fetchone()[0]
            item_count_mismatch=conn.execute('''SELECT COUNT(*) FROM p LEFT JOIN (SELECT pid,COUNT(*) c FROM i GROUP BY pid) s ON p.pid=s.pid WHERE p.reported_items!=COALESCE(s.c,0)''').fetchone()[0]
            cross_currency=conn.execute('''SELECT COUNT(*) FROM (SELECT pid FROM i WHERE pid IS NOT NULL AND currency IS NOT NULL GROUP BY pid HAVING COUNT(DISTINCT currency)>1)''').fetchone()[0]
            currency_details=conn.execute('''SELECT i.pid, GROUP_CONCAT(DISTINCT i.currency), COUNT(*) FROM i WHERE i.currency IS NOT NULL GROUP BY i.pid HAVING COUNT(DISTINCT i.currency)>1 ORDER BY i.pid''').fetchall()
            mismatch_examples=conn.execute('''SELECT p.pid,p.reported_items,COALESCE(s.c,0) FROM p LEFT JOIN (SELECT pid,COUNT(*) c FROM i GROUP BY pid) s ON p.pid=s.pid WHERE p.reported_items!=COALESCE(s.c,0) LIMIT ?''',(args.max_review,)).fetchall()
    for pid, currencies, cnt in currency_details[:args.max_review]:
        problems.append({'kind':'mixed_item_currencies','procedure_id':pid,'source_id':'','detail':f'{currencies}; items={cnt}'})
    for pid,expected,actual in mismatch_examples:
        problems.append({'kind':'item_count_mismatch','procedure_id':pid,'source_id':'','detail':f'expected={expected}; actual={actual}'})
    summary={'input_files':[str(procpath),str(itempath)],'counts':counts,
      'integrity':{'duplicate_procedure_id':pcounts['duplicate_procedure_id'],'missing_procedure_id':pcounts['missing_id'],'items_without_procedure':orphans,'duplicate_source_record_ids':unique_ids,'missing_source_record_id':icounts['missing_source_record_id'],'item_count_mismatches':item_count_mismatch,'procedures_with_multiple_item_currencies':cross_currency},
      'quality_warnings':dict(warnings),'null_counts':{k:dict(v) for k,v in nulls.items()},
      'categories_top':cats.most_common(25),'procurement_types_top':types.most_common(25),'procedures_by_year':dict(years),'items_by_year':dict(annual),'item_currencies':dict(money),
      'tech_candidate_examples_exported':len(tech_candidates),
      'notes':['Coincidencias TI: heuristica de palabras clave, NO etiquetas verificadas.','Los importes provienen del ETL; no son convertidos a una moneda comun.','La discrepancia de monedas a nivel de items no significa necesariamente que la fuente sea incorrecta.']}
    write_json(args.output/'audit_summary.json',summary)
    csv_out(args.output/'integrity_review.csv',['kind','procedure_id','source_id','detail'],problems)
    csv_out(args.output/'tech_candidates_review.csv',['procedure_id','source_year','title','category','reason','label_status'],tech_candidates)
    csv_out(args.output/'field_completeness.csv',['table','field','rows','missing','missing_pct'],[
      {'table':table,'field':field,'rows':counts[table],'missing':missing,'missing_pct':round(100*missing/max(1,counts[table]),4)}
      for table,fields in nulls.items() for field,missing in sorted(fields.items())])
    lines=['# Auditoria CONOSCE','',f"Procedimientos: **{counts['procedures']:,}**; items: **{counts['procedure_items']:,}**.",'',
    '## Integridad','']+[f'- {k}: **{v:,}**' for k,v in summary['integrity'].items()]+['','## Advertencias de calidad','']+[f'- {k}: {v:,}' for k,v in warnings.items()]+['',
    '## Interpretacion','Los candidatos TI se extraen por heuristica de palabras clave y requieren revision humana.','No confundir registro de item con procedimiento independiente.']
    (args.output/'audit_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Auditoria completada:', {'procedures':counts['procedures'],'items':counts['procedure_items'],'integrity':summary['integrity'],'output':str(args.output)})

if __name__=='__main__':main()
