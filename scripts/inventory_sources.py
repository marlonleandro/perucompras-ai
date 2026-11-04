#!/usr/bin/env python3
"""Inventario reproducible de fuentes CONOSCE y bases estandar.

Escanea archivos locales (--raw) o interpreta una lista de Windows DIR (--listing).
No modifica las fuentes originales. Python 3.10+ sin dependencias externas.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile, BadZipFile

MONTHS = {name: i + 1 for i, name in enumerate(['ENERO','FEBRERO','MARZO','ABRIL','MAYO','JUNIO','JULIO','AGOSTO','SEPTIEMBRE','OCTUBRE','NOVIEMBRE','DICIEMBRE'])}
FAMILIES = ['CONVOCATORIAS','ADJUDICACIONES','CONTRATOS','CONSORCIO','PROVEEDORES','ORDENESCOMPRA']
CANDIDATE_ALIASES = {
    'procedure_id': ['ID PROCEDIMIENTO','ID_PROCEDIMIENTO','CODIGO PROCEDIMIENTO','CODIGO DE PROCEDIMIENTO','NOMENCLATURA','NRO PROCEDIMIENTO','ID CONVOCATORIA'],
    'source_record_id': ['ID','ID REGISTRO','ID_REGISTRO','CODIGO','CODIGO UNICO'],
    'entity_name': ['ENTIDAD','NOMBRE ENTIDAD','NOMBRE DE LA ENTIDAD','ENTIDAD CONVOCANTE','RAZON SOCIAL ENTIDAD'],
    'title': ['OBJETO DE CONTRATACION','OBJETO DE LA CONTRATACION','DESCRIPCION DEL OBJETO','OBJETO','DENOMINACION'],
    'description': ['DESCRIPCION','DETALLE','DESCRIPCION DEL PROCEDIMIENTO'],
    'procurement_type': ['TIPO DE PROCEDIMIENTO','TIPO PROCEDIMIENTO','TIPO DE CONTRATACION'],
    'category': ['TIPO DE OBJETO','OBJETO CONTRACTUAL','CATEGORIA','TIPO DE OBJETO CONTRACTUAL'],
    'publication_date': ['FECHA DE CONVOCATORIA','FECHA CONVOCATORIA','FECHA PUBLICACION','FECHA DE PUBLICACION'],
    'estimated_value': ['VALOR ESTIMADO','VALOR REFERENCIAL','MONTO REFERENCIAL','MONTO ESTIMADO'],
    'currency': ['MONEDA','TIPO MONEDA'],
    'status': ['ESTADO DEL PROCEDIMIENTO','ESTADO PROCEDIMIENTO','ESTADO'],
    'legal_regime': ['REGIMEN LEGAL','NORMATIVA APLICABLE','LEY APLICABLE'],
    'source_url': ['URL','ENLACE','LINK','URL PROCEDIMIENTO'],
}
FIELDS = list(CANDIDATE_ALIASES) + ['retrieved_at']

def norm(s: str) -> str:
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii','ignore').decode('ascii')
    return re.sub(r'[^A-Z0-9]+',' ',s.upper()).strip()

def classify(filename: str) -> dict:
    upper = norm(filename)
    out = dict(family='DESCONOCIDO', source_type='OTRO', year=None, month=None, part=None,
               priority='P2', expected_use='REVISAR', notes=[])
    original = filename.upper()
    if original.startswith('CONOSCE_'):
        out['source_type'] = 'CONOSCE_TABULAR'
        out['family'] = next((f for f in FAMILIES if f in original), 'CONOSCE_OTROS')
        out['priority'] = 'P0' if out['family'] in ('CONVOCATORIAS','ADJUDICACIONES') else 'P1'
        out['expected_use'] = {
            'CONVOCATORIAS':'Clasificacion y extraccion de procedimientos',
            'ADJUDICACIONES':'Resultados y trazabilidad',
            'CONTRATOS':'Vinculacion contractual',
            'CONSORCIO':'Participantes de consorcios',
            'PROVEEDORES':'Contexto de proveedores',
            'ORDENESCOMPRA':'Ordenes y adquisiciones',
        }.get(out['family'],'Revisar estructura')
        m = re.search(r'(202[0-9])(?:_(\d+))?\.XLSX$', original)
        if m:
            out['year'] = int(m.group(1))
            out['part'] = int(m.group(2)) if m.group(2) else None
        if out['family'] == 'ORDENESCOMPRA':
            for month, month_num in MONTHS.items():
                if month in original:
                    out['month'] = month_num
                    break
            if out['month'] is None:
                out['notes'].append('No se pudo detectar mes de ordenes de compra')
    elif 'BASES' in upper or 'EXPRESION DE INTERES' in upper:
        out.update(source_type='DOCUMENTO_NORMATIVO',family='BASES_ESTANDAR',priority='P0',
                   expected_use='RAG normativo y ejemplos supervisados revisados')
        if filename.lower().endswith('.docx.docx'):
            out['notes'].append('Doble extension .docx.docx; verificar formato real antes de renombrar')
    return out

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def excel_metadata(path: Path):
    """Lee nombres de hojas y cabeceras sin cargar millones de filas. Solo XML de XLSX."""
    sheets=[]; columns=[]; errors=[]
    try:
        with ZipFile(path) as z:
            namespaces={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
                        'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
            wb=ET.fromstring(z.read('xl/workbook.xml'))
            sheet_entries=wb.find('m:sheets',namespaces)
            if sheet_entries is None:
                return sheets,columns,['No hay lista de hojas XLSX']
            sheets=[s.get('name','') for s in sheet_entries]
            # No lee sharedStrings: pueden ser enormes. Solo hojas, esquema pendiente.
    except (OSError, BadZipFile, ET.ParseError, KeyError) as e:
        errors.append('Error leyendo XLSX: '+str(e))
    return sheets,columns,errors

def read_listing(path: Path):
    raw=path.read_bytes()
    content=None
    for enc in ('utf-8-sig','cp850','cp1252','latin-1'):
        try: content=raw.decode(enc); break
        except UnicodeDecodeError: continue
    pattern=re.compile(r'^\s*(\d{2}/\d{2}/\d{4})\s+(\d{1,2}:\d{2}\s*[AP]M)\s+([\d,]+)\s+(.+?)\s*$')
    rows=[]
    for line in content.splitlines():
        match=pattern.match(line)
        if not match: continue
        date,time,size,name=match.groups()
        if name.lower() == 'lista_archivos.txt': continue
        rows.append(dict(name=name,bytes=int(size.replace(',','')),modified=date,path=None,exists=False))
    return rows

def read_directory(raw: Path):
    return [dict(name=p.name,bytes=p.stat().st_size,modified=datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                 path=p,exists=True) for p in sorted(raw.rglob('*')) if p.is_file() and not p.name.startswith('~$')]

def csv_save(path,rows,columns):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description='Inventario de fuentes PeruCompras AI')
    src=parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--raw',type=Path,help='Directorio local data/raw')
    src.add_argument('--listing',type=Path,help='Archivo de texto con DIR de Windows')
    parser.add_argument('--output',type=Path,default=Path('data/inventory'))
    parser.add_argument('--hash',action='store_true',help='Calcular SHA256 (puede tardar por el volumen)')
    parser.add_argument('--inspect-xlsx',action='store_true',help='Leer nombres de hojas XLSX sin cargar registros')
    args=parser.parse_args()
    if args.raw and not args.raw.is_dir(): parser.error(f'No existe directorio: {args.raw}')
    if args.listing and not args.listing.is_file(): parser.error(f'No existe lista: {args.listing}')
    source=read_directory(args.raw) if args.raw else read_listing(args.listing)
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]; sheet_rows=[]
    now=datetime.now(timezone.utc).isoformat()
    for i,item in enumerate(source,1):
        parsed=classify(item['name']); path=item['path']; notes=list(parsed['notes'])
        sheets=[]
        if args.inspect_xlsx and path and path.suffix.lower()=='.xlsx':
            sheets, _, errors=excel_metadata(path)
            notes+=errors
            for sheet in sheets: sheet_rows.append(dict(file_name=item['name'],sheet_name=sheet,header_status='NO_INSPECCIONADO'))
        r=dict(source_id=f'SRC-{i:04d}',filename=item['name'],relative_path=(str(path.relative_to(args.raw)) if path else item['name']),
               file_extension=''.join(Path(item['name']).suffixes[-2:]) if item['name'].lower().endswith('.docx.docx') else Path(item['name']).suffix.lower(),
               file_size_bytes=item['bytes'],file_size_mb=round(item['bytes']/1024**2,3),last_modified=item['modified'],
               source_system='CONOSCE/OECE' if parsed['source_type']=='CONOSCE_TABULAR' else 'OECE/PENDIENTE_VERIFICAR',
               **{k:parsed[k] for k in ('source_type','family','year','month','part','priority','expected_use')},
               source_url='',license='',license_verified='NO',sha256=sha256(path) if args.hash and path else '',
               actual_file_available='SI' if path else 'NO',sheet_count=len(sheets) if sheets else '',
               status='INVENTARIADO_METADATA' if path else 'SOLO_LISTADO',
               notes=' | '.join(notes),cataloged_at=now)
        rows.append(r)
    columns=list(rows[0]) if rows else ['source_id','filename']
    csv_save(args.output/'sources_inventory.csv',rows,columns)
    if args.inspect_xlsx: csv_save(args.output/'sheets_inventory.csv',sheet_rows,['file_name','sheet_name','header_status'])
    coverage=defaultdict(lambda: dict(files=0,total_bytes=0))
    for r in rows:
        key=(r['family'],r['year'] or 'SIN_ANIO',r['month'] or 'SIN_MES')
        coverage[key]['files']+=1;coverage[key]['total_bytes']+=r['file_size_bytes']
    coverage_rows=[dict(family=k[0],year=k[1],month=k[2],files=v['files'],total_bytes=v['total_bytes']) for k,v in sorted(coverage.items(),key=lambda x:tuple(map(str,x[0])))]
    csv_save(args.output/'coverage.csv',coverage_rows,['family','year','month','files','total_bytes'])
    mapping=[dict(canonical_field=f,description='Campo objetivo; no inferir del nombre del archivo',
                  candidate_headers=' | '.join(CANDIDATE_ALIASES.get(f,[])),
                  mapped_header='',mapping_status='PENDIENTE_REVISAR_XLSX',transformation_rule='',nullable='SI') for f in FIELDS]
    csv_save(args.output/'canonical_mapping_template.csv',mapping,list(mapping[0]))
    summary=Counter(r['family'] for r in rows)
    total_bytes=sum(r['file_size_bytes'] for r in rows)
    report=['# Inventario de fuentes — PerúCompras AI','',f'- Modalidad: {"escaneo real" if args.raw else "listado Windows DIR"}',
            f'- Archivos inventariados: **{len(rows)}**',f'- Tamaño total declarado: **{total_bytes/1024**3:.3f} GiB**',
            '- Contenido tabular y documentos: **no analizados** en modo listado.','',
            '## Archivos por familia','']
    report.extend(f'- {k}: {v}' for k,v in sorted(summary.items()))
    report+=['','## Advertencias','',
             '- `year` y `month` se deducen **solo del nombre** del archivo; no prueban cobertura interna.',
             '- `source_url`, licencia y régimen legal **no se inventan**. Deben validarse contra la descarga.',
             '- El mapeo canónico es una plantilla: no constituye una extracción de registros.',
             '- No aplicar el régimen legal por año sin revisar el procedimiento y fuente.',
             '- Cualquier archivo .docx.docx requiere comprobar que se abre correctamente.',
             '- Para construir el dataset, inspeccionar cabeceras y luego diseñar mapeos específicos por familia.',
             '- `retrieved_at` corresponde a la fecha real de descarga, no a la fecha de modificación local.','']
    (args.output/'inventory_report.md').write_text('\n'.join(report),encoding='utf-8')
    (args.output/'inventory_summary.json').write_text(json.dumps({'files':len(rows),'total_bytes':total_bytes,'families':dict(summary),'mode':'raw' if args.raw else 'listing'},indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'OK: {len(rows)} archivos, {total_bytes/1024**3:.3f} GiB. Resultados en {args.output}')
    print('Familias:',dict(summary))

if __name__=='__main__': main()
