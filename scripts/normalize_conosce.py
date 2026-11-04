#!/usr/bin/env python3
"""Normaliza CONOSCE_CONVOCATORIASYYYY_*.xlsx a procedimientos e items Parquet.

Lectura XLSX incremental con zipfile/XML (sin cargar Excel en memoria).
Dependencia para escribir Parquet: pyarrow. No modifica las fuentes originales.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import unicodedata
from xml.etree import ElementTree as ET
from zipfile import ZipFile

MAIN = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG_REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
NS = {'m': MAIN, 'r': REL}
SOURCE_PATTERN = re.compile(r'^CONOSCE_CONVOCATORIAS(20\d{2})_\d+\.xlsx$', re.I)
COLUMN_NAMES = [
    'CODIGOENTIDAD', 'ENTIDAD_RUC', 'ENTIDAD', 'TIPOENTIDAD', 'CODIGOCONVOCATORIA',
    'DESCRIPCION_PROCESO', 'PROCESO', 'TIPOCOMPRA', 'OBJETOCONTRACTUAL',
    'SECTOR / NIVEL DE GOBIERNO', 'SISTEMA_CONTRATACION', 'TIPOPROCESOSELECCION',
    'MONTOREFERENCIAL', 'N_ITEM', 'DESCRIPCION_ITEM', 'UNIDAD_MEDIDA', 'ESTADOITEM',
    'PAQUETE', 'CODIGOITEM', 'ITEMCUBSO', 'MONTO_REFERENCIAL_ITEM', 'MONEDA',
    'DEPARTAMENTO_ITEM', 'PROVINCIA_ITEM', 'DISTRITO_ITEM', 'FECHA_CONVOCATORIA',
    'FECHAINTEGRACIONBASES', 'FECHAPRESENTACIONPROPUESTA'
]
ERROR_FIELDS = ['file','sheet','excel_row','field','raw_value','issue','severity']


def normalize_header(s: str) -> str:
    s = unicodedata.normalize('NFKD', str(s or ''))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^A-Z0-9]', '', s.upper())


def clean(value):
    if value is None:
        return None
    s = str(value).strip().replace('\u00a0', ' ')
    return s if s else None


def read_shared_strings(zf: ZipFile):
    path = 'xl/sharedStrings.xml'
    if path not in zf.namelist():
        return []
    result = []
    with zf.open(path) as stream:
        for _, elem in ET.iterparse(stream, events=('end',)):
            if elem.tag == f'{{{MAIN}}}si':
                result.append(''.join(t.text or '' for t in elem.iter(f'{{{MAIN}}}t')))
                elem.clear()
    return result


def workbook_sheets(zf: ZipFile):
    wb = ET.fromstring(zf.read('xl/workbook.xml'))
    rels = ET.fromstring(zf.read('xl/_rels/workbook.xml.rels'))
    links = {}
    for r in rels:
        rid = r.attrib.get('Id')
        target = r.attrib.get('Target', '')
        if target.startswith('/'):
            path = target.lstrip('/')
        else:
            import posixpath
            path = posixpath.normpath(posixpath.join('xl', target))
        links[rid] = path
    date1904 = False
    props = wb.find('m:workbookPr', NS)
    if props is not None:
        date1904 = props.attrib.get('date1904') in ('1', 'true', 'True')
    sheets = []
    for s in wb.findall('m:sheets/m:sheet', NS):
        rid = s.attrib.get(f'{{{REL}}}id')
        path = links.get(rid)
        if path and path in zf.namelist():
            sheets.append((s.attrib['name'], path))
    return sheets, date1904


def cell_value(cell, shared):
    t = cell.attrib.get('t')
    if t == 'inlineStr':
        return ''.join(n.text or '' for n in cell.iter(f'{{{MAIN}}}t'))
    v = cell.find(f'{{{MAIN}}}v')
    if v is None or v.text is None:
        return None
    if t == 's':
        try:
            return shared[int(v.text)]
        except (IndexError, ValueError):
            return None
    if t == 'b':
        return 'true' if v.text == '1' else 'false'
    if t == 'e':
        return None
    return v.text


def col_num(reference):
    m = re.match(r'([A-Z]+)', reference.upper())
    if not m:
        return -1
    v = 0
    for c in m.group(1):
        v = v * 26 + ord(c) - 64
    return v - 1


def iter_sheet(zf, path, shared):
    with zf.open(path) as stream:
        for event, elem in ET.iterparse(stream, events=('end',)):
            if elem.tag == f'{{{MAIN}}}row':
                r = int(elem.attrib.get('r', '0'))
                out = {}
                for cell in elem.findall(f'{{{MAIN}}}c'):
                    idx = col_num(cell.attrib.get('r', ''))
                    if idx >= 0:
                        out[idx] = clean(cell_value(cell, shared))
                yield r, out
                elem.clear()


def parse_amount(raw):
    if raw is None:
        return None
    s = str(raw).strip()
    s = re.sub(r'(?i)\b(?:S/\.?|PEN|USD|US\$|SOLES|DOLARES)\b', '', s)
    s = re.sub(r'[^\d.,()\-+]', '', s)
    negative = s.startswith('(') and s.endswith(')')
    s = s.strip('()')
    if not s or s in ('-', '+'):
        return None
    if ',' in s and '.' in s:
        decimal_sep = ',' if s.rfind(',') > s.rfind('.') else '.'
        thousands = '.' if decimal_sep == ',' else ','
        s = s.replace(thousands, '').replace(decimal_sep, '.')
    elif ',' in s or '.' in s:
        sep = ',' if ',' in s else '.'
        chunks = s.split(sep)
        if len(chunks) > 2 and all(len(c) == 3 for c in chunks[1:]):
            s = ''.join(chunks)
        elif len(chunks) == 2 and len(chunks[1]) == 3 and len(chunks[0]) <= 3:
            # 1,234 ambiguo: para montos CONOSCE asumir separador de millares.
            s = ''.join(chunks)
        elif len(chunks) == 2:
            s = chunks[0] + '.' + chunks[1]
        else:
            return None
    try:
        d = Decimal(s)
        if not d.is_finite():
            return None
        return str(-d if negative else d)
    except InvalidOperation:
        return None


def parse_date(raw, date1904=False):
    if raw is None:
        return None
    s = clean(raw)
    if not s:
        return None
    if re.fullmatch(r'\d{1,5}(?:\.\d+)?', s):
        try:
            days = float(s)
            if days < 1 or days > 100000:
                return None
            epoch = datetime(1904, 1, 1) if date1904 else datetime(1899, 12, 30)
            return (epoch + timedelta(days=days)).date().isoformat()
        except (ValueError, OverflowError):
            return None
    try:
        return datetime.fromisoformat(s.replace('Z','+00:00')).date().isoformat()
    except ValueError:
        pass
    for fmt in ('%d/%m/%Y','%d-%m-%Y','%Y/%m/%d','%Y-%m-%d',
                '%d/%m/%Y %H:%M:%S','%m/%d/%Y %H:%M:%S'):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def error(writer, counter, file, sheet, row, field, raw, issue, severity='warning'):
    writer.writerow({'file':file,'sheet':sheet,'excel_row':row,'field':field,
                    'raw_value':str(raw or '')[:300],'issue':issue,'severity':severity})
    counter[issue] += 1


def stage(root, db, err_writer, stats, strict=False, limit=0):
    files = sorted(p for p in root.glob('*.xlsx') if SOURCE_PATTERN.fullmatch(p.name))
    if not files:
        raise ValueError(f'No se encontraron CONOSCE_CONVOCATORIASYYYY_*.xlsx en {root}')
    for file in files:
        year = int(SOURCE_PATTERN.fullmatch(file.name).group(1))
        stats['files'].append(file.name)
        with ZipFile(file) as zf:
            shared = read_shared_strings(zf)
            sheets, date1904 = workbook_sheets(zf)
            for sheet, path in sheets:
                iterator = iter_sheet(zf, path, shared)
                try:
                    header_row, header = next(iterator)
                except StopIteration:
                    continue
                by_name = {normalize_header(v): i for i, v in header.items() if v}
                missing = [c for c in COLUMN_NAMES if normalize_header(c) not in by_name]
                if missing:
                    error(err_writer, stats['issues'], file.name, sheet, header_row, 'HEADER',
                          ','.join(missing), 'missing_columns', 'error')
                    if strict:
                        raise ValueError(f'{file.name}/{sheet}: faltan columnas {missing}')
                if 'CODIGOCONVOCATORIA' not in by_name:
                    continue
                for excel_row, cells in iterator:
                    if not any(cells.values()):
                        continue
                    stats['rows_seen'] += 1
                    def get(field):
                        idx = by_name.get(normalize_header(field))
                        return cells.get(idx) if idx is not None else None
                    code = get('CODIGOCONVOCATORIA')
                    if not code:
                        error(err_writer, stats['issues'], file.name, sheet, excel_row,
                              'CODIGOCONVOCATORIA','', 'missing_procedure_code', 'error')
                    procedure_id = f'{year}:{code}' if code else None
                    source_id = f'{file.name}:{sheet}:{excel_row}'
                    proc_amount = parse_amount(get('MONTOREFERENCIAL'))
                    item_amount = parse_amount(get('MONTO_REFERENCIAL_ITEM'))
                    for field, value, parsed in [
                        ('MONTOREFERENCIAL',get('MONTOREFERENCIAL'),proc_amount),
                        ('MONTO_REFERENCIAL_ITEM',get('MONTO_REFERENCIAL_ITEM'),item_amount)]:
                        if value is not None and parsed is None:
                            error(err_writer,stats['issues'],file.name,sheet,excel_row,
                                  field,value,'invalid_amount')
                    parsed_dates = {}
                    for field in ('FECHA_CONVOCATORIA','FECHAINTEGRACIONBASES','FECHAPRESENTACIONPROPUESTA'):
                        value = get(field)
                        parsed_dates[field] = parse_date(value, date1904)
                        if value and not parsed_dates[field]:
                            error(err_writer,stats['issues'],file.name,sheet,excel_row,
                                  field,value,'invalid_date')
                    data = {
                        'procedure_id':procedure_id,
                        'source_record_id':source_id,
                        'source_file':file.name,'source_sheet':sheet,'source_excel_row':excel_row,
                        'source_year':year,
                        'entity_code':get('CODIGOENTIDAD'),
                        'entity_ruc':get('ENTIDAD_RUC'),
                        'entity_name':get('ENTIDAD'),
                        'entity_type':get('TIPOENTIDAD'),
                        'title':get('DESCRIPCION_PROCESO'),
                        'procedure_label':get('PROCESO'),
                        'purchase_type':get('TIPOCOMPRA'),
                        'category':get('OBJETOCONTRACTUAL'),
                        'government_sector':get('SECTOR / NIVEL DE GOBIERNO'),
                        'contract_system':get('SISTEMA_CONTRATACION'),
                        'procurement_type':get('TIPOPROCESOSELECCION'),
                        'procedure_amount':proc_amount,
                        'procedure_amount_raw':get('MONTOREFERENCIAL'),
                        'item_number':get('N_ITEM'),
                        'item_code':get('CODIGOITEM'),
                        'item_description':get('DESCRIPCION_ITEM'),
                        'item_unit':get('UNIDAD_MEDIDA'),
                        'item_status':get('ESTADOITEM'),
                        'package':get('PAQUETE'),
                        'item_cubso':get('ITEMCUBSO'),
                        'item_amount':item_amount,
                        'item_amount_raw':get('MONTO_REFERENCIAL_ITEM'),
                        'currency_raw':get('MONEDA'),
                        'currency':normalize_currency(get('MONEDA')),
                        'department':get('DEPARTAMENTO_ITEM'),
                        'province':get('PROVINCIA_ITEM'),
                        'district':get('DISTRITO_ITEM'),
                        'publication_date':parsed_dates['FECHA_CONVOCATORIA'],
                        'bases_integration_date':parsed_dates['FECHAINTEGRACIONBASES'],
                        'proposal_date':parsed_dates['FECHAPRESENTACIONPROPUESTA'],
                        'legal_regime':None,
                        'source_url':None,
                        'retrieved_at':stats['retrieved_at'],
                    }
                    if data['currency_raw'] and data['currency'] is None:
                        error(err_writer,stats['issues'],file.name,sheet,excel_row,'MONEDA',
                              data['currency_raw'],'unknown_currency')
                    db.execute('INSERT INTO raw (procedure_id, source_id, item_number, item_code, payload) VALUES (?,?,?,?,?)',
                               (procedure_id,source_id,data['item_number'],data['item_code'],json.dumps(data,ensure_ascii=False)))
                    stats['rows_imported'] += 1
                    if stats['rows_imported'] % 2000 == 0:
                        db.commit()
                    if limit and stats['rows_imported'] >= limit:
                        db.commit()
                        return
        db.commit()


def normalize_currency(s):
    if s is None:
        return None
    s = unicodedata.normalize('NFKD',s.upper())
    s = ''.join(x for x in s if not unicodedata.combining(x)).strip()
    if s in ('PEN','S/','S/.','SOLES','SOL','NUEVOS SOLES','SOL PERUANO'):
        return 'PEN'
    if s in ('USD','US$','$','DOLARES','DOLAR AMERICANO','DOLARES AMERICANOS',
             'DOLAR NORTEAMERICANO','DOLARES NORTEAMERICANOS'):
        return 'USD'
    if s in ('EUR','EUROS','EURO','€'):
        return 'EUR'
    if s in ('GBP','LIBRA ESTERLINA','LIBRAS ESTERLINAS'):
        return 'GBP'
    if s in ('CAD','DOLAR CANADIENSE','DOLARES CANADIENSES'):
        return 'CAD'
    if s in ('CHF','FRANCO SUIZO','FRANCOS SUIZOS'):
        return 'CHF'
    if s in ('BRL','REAL BRASILENO','REALES BRASILENOS','REAL BRASILENO'):
        return 'BRL'
    return None


PROCEDURE_COLUMNS = [
  'procedure_id','source_year','entity_code','entity_ruc','entity_name','entity_type',
  'title','procedure_label','purchase_type','category','government_sector','contract_system',
  'procurement_type','estimated_value','estimated_value_raw','currency','publication_date',
  'bases_integration_date','proposal_date','legal_regime','source_url','retrieved_at',
  'source_file','source_sheet','source_excel_row','item_count'
]
ITEM_COLUMNS = [
  'source_record_id','procedure_id','source_year','item_number','item_code','description',
  'unit','status','package','item_cubso','estimated_value','estimated_value_raw','currency',
  'department','province','district','source_file','source_sheet','source_excel_row',
  'retrieved_at'
]


def build_outputs(db, out, err_writer, stats, output_format="parquet", batch_size=5000):
    if output_format == 'parquet':
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit('Falta pyarrow. Instala: python -m pip install pyarrow') from exc

    class BatchWriter:
        def __init__(self, basename, columns):
            self.columns=columns
            if output_format == 'parquet':
                schema=pa.schema([(c,pa.int64() if c in ('source_year','source_excel_row','item_count') else pa.string()) for c in columns])
                self.writer=pq.ParquetWriter(str(out/(basename+'.parquet')),schema,compression='snappy')
                self.schema=schema
            else:
                self.stream=(out/(basename+'.csv')).open('w',newline='',encoding='utf-8-sig')
                self.writer=csv.DictWriter(self.stream,fieldnames=columns)
                self.writer.writeheader()
        def write(self, records):
            if not records:
                return
            if output_format == 'parquet':
                self.writer.write_table(pa.Table.from_pylist(records,schema=self.schema))
            else:
                self.writer.writerows(records)
        def close(self):
            if output_format == 'parquet':
                self.writer.close()
            else:
                self.stream.close()

    ipw=BatchWriter('procedure_items',ITEM_COLUMNS)
    ppw=BatchWriter('procedures',PROCEDURE_COLUMNS)
    rows=[]
    for (payload,) in db.execute('SELECT payload FROM raw ORDER BY seq'):
        d=json.loads(payload)
        item={
            'source_record_id':d['source_record_id'],'procedure_id':d['procedure_id'],
            'source_year':d['source_year'],'item_number':d['item_number'],'item_code':d['item_code'],
            'description':d['item_description'],'unit':d['item_unit'],'status':d['item_status'],
            'package':d['package'],'item_cubso':d['item_cubso'],'estimated_value':d['item_amount'],
            'estimated_value_raw':d['item_amount_raw'],'currency':d['currency'],
            'department':d['department'],'province':d['province'],'district':d['district'],
            'source_file':d['source_file'],'source_sheet':d['source_sheet'],
            'source_excel_row':d['source_excel_row'],'retrieved_at':d['retrieved_at']
        }
        rows.append(item)
        if len(rows)>=batch_size:
            ipw.write(rows)
            rows=[]
    ipw.write(rows)
    ipw.close()

    # Una sola pasada agrupada: evita cargar todos los items de un procedimiento.
    rows=[]
    current_pid=None
    first=None
    item_count=0
    seen={}
    check_fields=['entity_ruc','entity_name','title','procurement_type','category',
                  'publication_date','procedure_amount','currency']

    def finish_group(pid, first, count, seen):
        if first is None:
            return
        for field, values in seen.items():
            if len(values)>1:
                error(err_writer,stats['issues'],first['source_file'],first['source_sheet'],
                      first['source_excel_row'],field,f'{len(values)} valores diferentes',
                      'procedure_conflict')
        rows.append({
            'procedure_id':pid,'source_year':first['source_year'],
            'entity_code':first['entity_code'],'entity_ruc':first['entity_ruc'],
            'entity_name':first['entity_name'],'entity_type':first['entity_type'],
            'title':first['title'],'procedure_label':first['procedure_label'],
            'purchase_type':first['purchase_type'],'category':first['category'],
            'government_sector':first['government_sector'],
            'contract_system':first['contract_system'],'procurement_type':first['procurement_type'],
            'estimated_value':first['procedure_amount'],'estimated_value_raw':first['procedure_amount_raw'],
            'currency':first['currency'],'publication_date':first['publication_date'],
            'bases_integration_date':first['bases_integration_date'],
            'proposal_date':first['proposal_date'],'legal_regime':None,'source_url':None,
            'retrieved_at':first['retrieved_at'],'source_file':first['source_file'],
            'source_sheet':first['source_sheet'],'source_excel_row':first['source_excel_row'],
            'item_count':count
        })
        stats['procedures']+=1
        if len(rows)>=batch_size:
            ppw.write(rows)
            rows.clear()

    for pid,payload in db.execute('SELECT procedure_id,payload FROM raw WHERE procedure_id IS NOT NULL ORDER BY procedure_id,seq'):
        d=json.loads(payload)
        if pid != current_pid:
            finish_group(current_pid,first,item_count,seen)
            current_pid=pid
            first=d
            item_count=0
            seen={field:set() for field in check_fields}
        item_count+=1
        for field in check_fields:
            val=d.get(field)
            if val is not None:
                seen[field].add(str(val))
    finish_group(current_pid,first,item_count,seen)
    ppw.write(rows)
    ppw.close()
    stats['items']=stats['rows_imported']


def main(argv=None):
    parser=argparse.ArgumentParser(description='Normaliza convocatorias CONOSCE (XLSX) a Parquet.')
    parser.add_argument('--raw',type=Path,default=Path('data/raw'))
    parser.add_argument('--output',type=Path,default=Path('data/processed'))
    parser.add_argument('--limit',type=int,default=0,help='Limita filas importadas para smoke test')
    parser.add_argument('--format',choices=('parquet','csv'),default='parquet',help='Formato de salida (default: parquet)')
    parser.add_argument('--strict',action='store_true',help='Fallar si faltan columnas')
    args=parser.parse_args(argv)
    if args.limit < 0:
        parser.error('--limit debe ser >= 0')
    args.output.mkdir(parents=True,exist_ok=True)
    errors_path=args.output/'normalization_errors.csv'
    stats={'files':[],'rows_seen':0,'rows_imported':0,'items':0,'procedures':0,
           'issues':Counter(),'retrieved_at':datetime.now(timezone.utc).isoformat()}
    with tempfile.TemporaryDirectory(prefix='conosce_') as td:
        db=sqlite3.connect(str(Path(td)/'stage.sqlite'))
        db.execute('CREATE TABLE raw(seq INTEGER PRIMARY KEY AUTOINCREMENT, procedure_id TEXT, source_id TEXT, item_number TEXT, item_code TEXT, payload TEXT)')
        db.execute('CREATE INDEX ix_procedure ON raw(procedure_id)')
        with errors_path.open('w',encoding='utf-8-sig',newline='') as stream:
            ew=csv.DictWriter(stream,fieldnames=ERROR_FIELDS)
            ew.writeheader()
            stage(args.raw,db,ew,stats,strict=args.strict,limit=args.limit)
            build_outputs(db,args.output,ew,stats,output_format=args.format)
        db.close()
    stats['issues']=dict(stats['issues'])
    stats['output_files']=[f'procedures.{args.format}',f'procedure_items.{args.format}','normalization_errors.csv','normalization_summary.json']
    with (args.output/'normalization_summary.json').open('w',encoding='utf-8') as stream:
        json.dump(stats,stream,ensure_ascii=False,indent=2)
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':
    sys.exit(main())
