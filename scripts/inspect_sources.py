#!/usr/bin/env python3
"""Auditoría de archivos XLSX CONOSCE sin cargar libros completos en memoria.

Python 3.10+, solo biblioteca estándar. Lee XML de los XLSX originales en modo streaming.
No modifica ni extrae los archivos de entrada.

Modo quick: inspecciona una muestra de filas por hoja, estima nulos sobre la muestra.
Modo full: recorre todas las filas, calcula nulos exactos de celdas no vacías.
Los conteos de `dimension` son metadatos declarados por Excel (no auditados).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from posixpath import normpath
from urllib.parse import unquote
from xml.etree import ElementTree as ET
from zipfile import ZipFile, BadZipFile

MAIN = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL_DOC = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL_PKG = 'http://schemas.openxmlformats.org/package/2006/relationships'
N = '{' + MAIN + '}'

# Correspondencias orientativas: la selección final requiere revisión del diccionario OECE.
ALIASES = {
    'procedure_id': ['ID PROCEDIMIENTO', 'CODIGO PROCEDIMIENTO', 'CODIGO PROCESO', 'NOMENCLATURA', 'ID CONVOCATORIA', 'IDENTIFICADOR PROCEDIMIENTO'],
    'source_record_id': ['ID REGISTRO', 'CODIGO UNICO', 'ID ADJUDICACION', 'ID CONTRATO', 'ID ORDEN', 'ID'],
    'entity_name': ['NOMBRE ENTIDAD', 'ENTIDAD CONVOCANTE', 'RAZON SOCIAL ENTIDAD', 'ENTIDAD', 'NOMBRE DE LA ENTIDAD'],
    'title': ['OBJETO DE CONTRATACION', 'OBJETO DE LA CONTRATACION', 'DESCRIPCION DEL OBJETO', 'DENOMINACION', 'OBJETO CONTRACTUAL'],
    'description': ['DESCRIPCION DEL PROCEDIMIENTO', 'DESCRIPCION', 'DETALLE DEL OBJETO', 'DESCRIPCION DEL OBJETO'],
    'procurement_type': ['TIPO DE PROCEDIMIENTO', 'TIPO PROCEDIMIENTO', 'TIPO DE CONTRATACION', 'TIPO DE PROCESO'],
    'category': ['OBJETO DE CONTRATACION', 'OBJETO CONTRACTUAL', 'TIPO DE OBJETO', 'CATEGORIA'],
    'publication_date': ['FECHA DE CONVOCATORIA', 'FECHA CONVOCATORIA', 'FECHA PUBLICACION', 'FECHA DE PUBLICACION'],
    'estimated_value': ['VALOR ESTIMADO', 'VALOR REFERENCIAL', 'MONTO REFERENCIAL', 'MONTO ESTIMADO'],
    'currency': ['TIPO MONEDA', 'MONEDA'],
    'status': ['ESTADO DEL PROCEDIMIENTO', 'ESTADO PROCEDIMIENTO', 'ESTADO'],
    'legal_regime': ['REGIMEN LEGAL', 'NORMATIVA APLICABLE', 'LEY APLICABLE'],
    'source_url': ['URL PROCEDIMIENTO', 'URL', 'ENLACE', 'LINK'],
    'retrieved_at': [],
}
MONTHS = {'ENERO':1,'FEBRERO':2,'MARZO':3,'ABRIL':4,'MAYO':5,'JUNIO':6,'JULIO':7,'AGOSTO':8,'SEPTIEMBRE':9,'OCTUBRE':10,'NOVIEMBRE':11,'DICIEMBRE':12}
FAMILIES = ['CONVOCATORIAS','ADJUDICACIONES','CONTRATOS','CONSORCIO','PROVEEDORES','ORDENESCOMPRA']
DATE_KEYS = ('FECHA', 'DATE', 'FEC ')


def norm(s):
    return re.sub(r'[^A-Z0-9]+',' ', unicodedata.normalize('NFKD', str(s)).encode('ascii','ignore').decode('ascii').upper()).strip()


def family_of(path):
    name=norm(path.name)
    if 'CONOSCE' in name:
        family=next((f for f in FAMILIES if f in name),'CONOSCE_OTROS')
        year = re.search(r'202[0-9]', name)
        month = next((v for k,v in MONTHS.items() if k in name), None) if family == 'ORDENESCOMPRA' else None
        return family, year.group() if year else '', month
    if 'BASES' in name or 'EXPRESION DE INTERES' in name:
        return 'BASES_ESTANDAR','',None
    return 'OTROS','',None


def xml_strings(el):
    # Preserva la concatenación de runs rich-text, pero evita textos de otros nodos.
    return ''.join(x.text or '' for x in el.iter() if x.tag == N+'t')


def workbook_sheets(z):
    """Retorna (nombre, ruta de XML) con resolución real de relationship ids."""
    wb=ET.fromstring(z.read('xl/workbook.xml'))
    relroot=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    relationships={r.attrib['Id']:r.attrib.get('Target','') for r in relroot.findall('{'+REL_PKG+'}Relationship')}
    result=[]
    for s in wb.findall('.//'+N+'sheets/'+N+'sheet'):
        name=s.attrib.get('name','')
        rid=s.attrib.get('{'+REL_DOC+'}id')
        target=relationships.get(rid,'')
        if not target:
            raise ValueError(f'Falta relación de hoja: {name}')
        # Nunca aceptar rutas fuera del zip previsto.
        if target.startswith('/'):
            part=target.lstrip('/')
        elif target.startswith('xl/'):
            part=target
        else:
            part=normpath('xl/'+target)
        part=unquote(part)
        if part.startswith('../') or part not in z.namelist():
            raise ValueError(f'Ruta de hoja inválida: {part}')
        result.append((name,part))
    return result


def column_index(cell_ref):
    m=re.match(r'([A-Z]+)', (cell_ref or '').upper())
    if not m:
        return None
    val=0
    for c in m.group(1):
        val=val*26+ord(c)-64
    return val-1


def excel_col(idx):
    n=idx+1; s=''
    while n:
        n,r=divmod(n-1,26); s=chr(65+r)+s
    return s


def cells_from_row(row):
    cells={}
    for cell in row.findall(N+'c'):
        idx=column_index(cell.attrib.get('r'))
        if idx is None: continue
        kind=cell.attrib.get('t','n')
        if kind=='inlineStr':
            el=cell.find(N+'is'); val=xml_strings(el) if el is not None else ''
        else:
            val_el=cell.find(N+'v')
            val=val_el.text if val_el is not None and val_el.text is not None else ''
        if val != '': cells[idx]=(kind,val)
    return cells


def worksheet_rows(z, part, max_rows=None):
    """Itera filas XML una a una, sin cargar el árbol completo."""
    with z.open(part) as stream:
        context=ET.iterparse(stream,events=('start','end'))
        _,root=next(context)
        count=0
        for event, elem in context:
            if event=='end' and elem.tag==N+'row':
                row_num=int(elem.attrib.get('r',str(count+1)))
                cells=cells_from_row(elem)
                yield row_num, cells
                elem.clear()
                root.clear()
                count+=1
                if max_rows is not None and count>=max_rows:
                    break


def worksheet_dimension(z, part):
    try:
        with z.open(part) as stream:
            for event,elem in ET.iterparse(stream,events=('start',)):
                if elem.tag==N+'dimension':
                    return elem.attrib.get('ref','')
                if elem.tag==N+'sheetData':
                    return ''
    except ET.ParseError:
        pass
    return ''


def load_selected_strings(z, indices):
    """Extrae solo los shared strings referidos por filas muestreadas."""
    if not indices or 'xl/sharedStrings.xml' not in z.namelist():
        return {}
    found={}; i=-1
    with z.open('xl/sharedStrings.xml') as stream:
        context=ET.iterparse(stream,events=('start','end'))
        _,root=next(context)
        for event, elem in context:
            if event=='end' and elem.tag==N+'si':
                i+=1
                if i in indices:
                    found[i]=xml_strings(elem)
                    if len(found)==len(indices): break
                elem.clear(); root.clear()
    return found


def decode(kind_val, shared):
    kind,val=kind_val
    if kind=='s':
        try: return shared.get(int(val),f'[shared-string:{val}]')
        except ValueError: return '[shared-string-invalid]'
    if kind=='b': return 'TRUE' if val=='1' else 'FALSE'
    return val


def infer_type(raw):
    kind,val=raw
    if kind in ('s','inlineStr','str'): return 'texto'
    if kind=='b': return 'booleano'
    if kind=='e': return 'error'
    if kind=='d': return 'fecha_iso'
    if kind=='n':
        try:
            num=float(val)
            return 'entero' if num.is_integer() else 'decimal'
        except ValueError: return 'desconocido'
    return 'desconocido'


def maybe_date(raw, name, shared, date1904=False):
    if not any(t in norm(name) for t in DATE_KEYS):
        return None
    kind,value=raw
    v=decode(raw,shared).strip()
    if not v: return None
    for fmt in ('%Y-%m-%d','%Y-%m-%d %H:%M:%S','%d/%m/%Y','%m/%d/%Y','%d-%m-%Y'):
        try:
            dt=datetime.strptime(v[:19],fmt)
            if 1990 <= dt.year <= 2100: return dt.date().isoformat()
        except ValueError: pass
    # Excel guarda fechas como número serial. Se infiere SOLO si el nombre de columna es fecha.
    if kind=='n':
        try:
            days=float(value)
            if 30000 < days < 75000:
                dt=datetime(1904,1,1) + timedelta(days=days) if date1904 else datetime(1899,12,30)+timedelta(days=days)
                return dt.date().isoformat()
        except (ValueError,OverflowError): pass
    return None


def choose_header(sampled):
    """Escoge fila de cabecera de las primeras 15 filas por densidad y texto."""
    choices=[]
    for num,cells in sampled[:15]:
        if not cells: continue
        count=len(cells)
        textual=sum(1 for kind,_ in cells.values() if kind in ('s','inlineStr','str'))
        choices.append((count+0.4*textual, -num, num, cells))
    if not choices:return None,{}
    _,_,num,cells=max(choices)
    return num,cells


def inspect_sheet(z, name, part, mode, sample_limit, date1904):
    # Sample up to 15 candidate header rows plus N data rows. Full mode reads twice.
    limit=15+sample_limit
    sampled=list(worksheet_rows(z,part,max_rows=limit))
    header_row,header_cells=choose_header(sampled)
    if header_row is None:
        return {'sheet':name,'sheet_part':part,'header_row':None,'columns':[],'summary':{'status':'empty'}}
    sample_data=[(r,c) for r,c in sampled if r>header_row][:sample_limit]
    used_indices={int(v) for _,cells in [(header_row,header_cells)]+sample_data for k,v in cells.values() if k=='s' and v.isdigit()}
    shared=load_selected_strings(z,used_indices)
    headers={idx:decode(kv,shared).strip().replace('\n',' ') for idx,kv in header_cells.items()}
    headers={idx:(val or f'SIN_NOMBRE_{excel_col(idx)}') for idx,val in headers.items()}
    # Duplicate headers distinguished by Excel column in outputs.
    if not headers:
        return {'sheet':name,'sheet_part':part,'header_row':header_row,'columns':[],'summary':{'status':'no_headers'}}
    types=defaultdict(Counter); notnull=Counter(); min_dates={}; max_dates={}
    examples=defaultdict(list)
    processed=0
    # Los nulos exactos en modo full se obtienen recorriendo todas las filas debajo del encabezado.
    if mode=='full':
        row_iter=worksheet_rows(z,part)
    else:
        row_iter=iter(sample_data)
    for row_num,cells in row_iter:
        if row_num<=header_row:continue
        if mode=='full' and not cells:continue
        processed+=1
        for idx,raw in cells.items():
            if idx not in headers:continue
            notnull[idx]+=1
            types[idx][infer_type(raw)]+=1
            if len(examples[idx])<3:
                val=decode(raw,shared)
                # Evitar volcar datos sensibles completos (RUC, nombres, DNI) en reporte.
                examples[idx].append(val[:40] if not val.startswith('[shared-string:') else '[texto compartido, no cargado]')
            dt=maybe_date(raw,headers[idx],shared,date1904)
            if dt:
                if idx not in min_dates or dt<min_dates[idx]:min_dates[idx]=dt
                if idx not in max_dates or dt>max_dates[idx]:max_dates[idx]=dt
    columns=[]
    for idx,name_col in sorted(headers.items()):
        matched=[]
        nrm=norm(name_col)
        for target,aliases in ALIASES.items():
            if nrm and nrm in (norm(a) for a in aliases):matched.append(target)
        probable=bool(matched)
        columns.append({
            'sheet':name,'excel_column':excel_col(idx),'source_column':name_col,
            'normalized_column':nrm,'canonical_candidates':'|'.join(matched),
            'candidate_status':'REVISAR' if probable else 'SIN_MAPEO',
            'type_observed':types[idx].most_common(1)[0][0] if types[idx] else 'sin_datos',
            'nonnull_rows':notnull[idx],
            'null_rows':max(0,processed-notnull[idx]),
            'null_pct':round(100*(processed-notnull[idx])/processed,2) if processed else None,
            'min_date_if_detected':min_dates.get(idx,''),
            'max_date_if_detected':max_dates.get(idx,''),
            'alias_warning':'MULTIPLE_CANDIDATOS' if len(matched)>1 else ''
        })
    return {'sheet':name,'sheet_part':part,'header_row':header_row,'columns':columns,
            'summary':{'status':'ok','rows_analyzed':processed,
                       'row_count_kind':'exacto_en_filas_con_contenido' if mode=='full' else 'muestra',
                       'worksheet_dimension_declared':worksheet_dimension(z,part),
                       'columns_detected':len(columns),
                       'date_inference_note':'Fechas numéricas inferidas por nombre de columna; revisar estilos de Excel y formatos ambiguos.'}}


def inspect_file(path, mode, sample_limit):
    with ZipFile(path) as z:
        sheets=workbook_sheets(z)
        wb=ET.fromstring(z.read('xl/workbook.xml'))
        wbpr=wb.find(N+'workbookPr')
        date1904=bool(wbpr is not None and wbpr.attrib.get('date1904','0') in ('1','true','True'))
        for name,part in sheets:
            try:
                yield inspect_sheet(z,name,part,mode,sample_limit,date1904)
            except (ET.ParseError,ValueError,KeyError,IndexError) as exc:
                yield {'sheet':name,'sheet_part':part,'header_row':None,'columns':[],
                       'summary':{'status':'error','error':str(exc)}}


def csv_write(path, rows, fieldnames):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Inspección de XLSX CONOSCE sin cargar los libros completos.')
    parser.add_argument('--raw',type=Path,required=True,help='Carpeta data/raw que contiene XLSX y DOCX')
    parser.add_argument('--output',type=Path,default=Path('data/inventory/inspection'))
    parser.add_argument('--mode',choices=['quick','full'],default='quick',help='quick: muestra; full: todas las filas de cada hoja')
    parser.add_argument('--sample-rows',type=int,default=500,help='Filas de datos muestreadas en modo quick (default 500)')
    parser.add_argument('--family',default='',help='Filtro de familia, p. ej. CONVOCATORIAS')
    parser.add_argument('--limit-files',type=int,default=0,help='Limita cantidad de archivos para pruebas')
    parser.add_argument('--recursive',action='store_true',help='Recorre carpetas anidadas')
    args=parser.parse_args(argv)
    if args.sample_rows<1:parser.error('--sample-rows debe ser >= 1')
    if not args.raw.is_dir():parser.error(f'Carpeta no encontrada: {args.raw}')
    args.output.mkdir(parents=True,exist_ok=True)
    files=sorted(args.raw.rglob('*') if args.recursive else args.raw.iterdir(),key=lambda p:p.name.lower())
    files=[p for p in files if p.is_file() and p.suffix.lower() in ('.xlsx','.docx') and not p.name.startswith('~$')]
    if args.family:files=[p for p in files if family_of(p)[0]==args.family.upper()]
    if args.limit_files:files=files[:args.limit_files]
    file_rows=[];sheet_rows=[];col_rows=[];map_rows=[];errors=[]
    inspected=0
    for num,p in enumerate(files,1):
        fam,year,month=family_of(p)
        print(f'[{num}/{len(files)}] {p.name}',flush=True)
        meta={'file':p.name,'relative_path':str(p.relative_to(args.raw)),'family':fam,'year':year,
              'month':month or '', 'size_bytes':p.stat().st_size,'file_type':p.suffix.lower(),
              'sheets':0,'status':'documento_sin_inspeccion_tabular' if p.suffix.lower()=='.docx' else 'ok',
              'error':''}
        if p.suffix.lower()=='.xlsx':
            try:
                for result in inspect_file(p,args.mode,args.sample_rows):
                    s=result['summary'];inspected+=1;meta['sheets']+=1
                    sheet_rows.append({'file':p.name,'family':fam,'year':year,'sheet':result['sheet'],
                                       'header_row':result['header_row'],'mode':args.mode,
                                       'rows_analyzed':s.get('rows_analyzed',0),
                                       'row_count_kind':s.get('row_count_kind',''),
                                       'worksheet_dimension_declared':s.get('worksheet_dimension_declared',''),
                                       'columns_detected':s.get('columns_detected',0),
                                       'status':s['status']})
                    if s['status']!='ok':
                        errors.append({'file':p.name,'sheet':result['sheet'],'error':s.get('error',s['status'])})
                    for col in result['columns']:
                        col_rows.append({'file':p.name,'family':fam,'year':year,**col})
                        map_rows.append({'file':p.name,'sheet':result['sheet'],
                                         'source_column':col['source_column'],
                                         'canonical_candidates':col['canonical_candidates'],
                                         'approved_target':'', 'mapping_rule':'',
                                         'approval_status':'PENDIENTE_VALIDACION',
                                         'reference_dictionary':'', 'notes':''})
            except (BadZipFile, OSError, ValueError, KeyError, ET.ParseError) as exc:
                meta['status']='error';meta['error']=str(exc)
                errors.append({'file':p.name,'sheet':'','error':str(exc)})
        file_rows.append(meta)
    csv_write(args.output/'files_inspected.csv',file_rows,
              ['file','relative_path','family','year','month','size_bytes','file_type','sheets','status','error'])
    csv_write(args.output/'sheets_profile.csv',sheet_rows,
              ['file','family','year','sheet','header_row','mode','rows_analyzed','row_count_kind','worksheet_dimension_declared','columns_detected','status'])
    csv_write(args.output/'columns_profile.csv',col_rows,
              ['file','family','year','sheet','excel_column','source_column','normalized_column','canonical_candidates','candidate_status','type_observed','nonnull_rows','null_rows','null_pct','min_date_if_detected','max_date_if_detected','alias_warning'])
    csv_write(args.output/'canonical_mapping_review.csv',map_rows,
              ['file','sheet','source_column','canonical_candidates','approved_target','mapping_rule','approval_status','reference_dictionary','notes'])
    csv_write(args.output/'inspection_errors.csv',errors,['file','sheet','error'])
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'mode':args.mode,
             'sample_rows':args.sample_rows if args.mode=='quick' else None,
             'files_seen':len(files),'xlsx_files':sum(r['file_type']=='.xlsx' for r in file_rows),
             'documents':sum(r['file_type']=='.docx' for r in file_rows),
             'sheets_inspected':inspected,'columns_detected':len(col_rows),
             'sheets_with_errors':len(errors),'families':dict(Counter(r['family'] for r in file_rows)),
             'limitations':['Modo quick: nulos y tipos se estiman sobre las primeras filas; no son cifras globales.',
                            'Fecha numérica inferida por encabezado, no se inspeccionaron estilos de fecha.',
                            'No se han validado llaves de unión ni duplicados entre fuentes.',
                            'Mapeos canónicos orientativos sujetos a diccionarios oficiales.',
                            'No se exportan ejemplos de celdas para prevenir exposición de datos personales.']}
    (args.output/'inspection_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Inspección de fuentes — PerúCompras AI','',
           f'- Modo: **{args.mode}**',f'- Archivos: **{len(files)}**',
           f'- Hojas XLSX inspeccionadas: **{inspected}**',
           f'- Columnas identificadas: **{len(col_rows)}**',
           f'- Errores: **{len(errors)}**','',
           '## Interpretación','',
           '- En modo quick los porcentajes de nulos y tipos solo corresponden a la muestra.',
           '- En modo full las filas con contenido son contabilizadas; no se incluye el encabezado.',
           '- Las fechas numéricas son inferidas y requieren validación de formato.',
           '- `canonical_mapping_review.csv` es una propuesta de correspondencia, **no un mapeo aprobado**.',
           '- Los archivos originales permanecen sin modificaciones.','',
           '## Siguiente paso','',
           'Revisar identificadores de procedimiento y relaciones entre archivos utilizando los diccionarios oficiales OECE.']
    (args.output/'inspection_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\nInspección terminada. Resultados en:',args.output)
    print(json.dumps({k:summary[k] for k in ('files_seen','sheets_inspected','columns_detected','sheets_with_errors')},ensure_ascii=False))
    return 0 if not errors else 2

if __name__=='__main__':
    sys.exit(main())
