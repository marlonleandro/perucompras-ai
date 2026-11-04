import csv
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import common_conosce as common
import build_sft_dataset as sft
import audit_conosce as audit

PROCS=[
 {'procedure_id':f'2025:{i}','source_year':2025,'entity_name':'Municipalidad Ficticia',
  'title':f'Servicio informatico ficticio {i}','category':'SERVICIOS' if i%2 else 'BIENES',
  'procurement_type':'CONCURSO PUBLICO','publication_date':'2025-01-02',
  'estimated_value':'1500.00','currency':'PEN','item_count':1,
  'source_file':'fixture.xlsx','source_sheet':'Sheet1','source_excel_row':i+1}
 for i in range(1,51)]
ITEMS=[{'procedure_id':p['procedure_id'],'source_record_id':f'fixture:{i}',
  'description':'Desarrollo de software' if i%2 else 'Licencias informaticas',
  'item_number':'1','item_code':str(i),'estimated_value':'1500.00','currency':'PEN',
  'source_file':'fixture.xlsx','source_excel_row':i+1,'source_year':2025}
 for i,p in enumerate(PROCS,1)]

class FakePQ:
 def ParquetFile(self,path):
  name=Path(path).name
  rows=PROCS if name=='procedures.parquet' else ITEMS
  return types.SimpleNamespace(schema_arrow=types.SimpleNamespace(names=list(rows[0].keys())))

def fake_iter(path,columns=None,batch_size=8192):
 rows=PROCS if Path(path).name=='procedures.parquet' else ITEMS
 for r in rows: yield {k:r.get(k) for k in columns} if columns else r.copy()

class PipelineTests(unittest.TestCase):
 def test_stable_split(self):
  self.assertEqual(common.choose_split('2025:900'),common.choose_split('2025:900'))
  self.assertIn(common.choose_split('2025:900'),{'train','val','test'})
 def test_labels_not_in_classification_prompt(self):
  p=PROCS[0]; item=ITEMS[0]
  x=dict(sft.make_examples(p,item))['classification']
  self.assertNotIn('SERVICIOS',x['input'])
  self.assertEqual(json.loads(x['output'])['category'],'SERVICIOS')
 def test_missing_values_not_invented(self):
  p=dict(PROCS[0],publication_date=None)
  obj=json.loads(dict(sft.make_examples(p,ITEMS[0]))['extraction']['output'])
  self.assertIsNone(obj['publication_date'])
 def test_full_sft_and_audit(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);inp=root/'processed';inp.mkdir()
   (inp/'procedures.parquet').touch();(inp/'procedure_items.parquet').touch()
   with patch.object(sft,'require_arrow',return_value=FakePQ()),patch.object(sft,'iter_parquet',side_effect=fake_iter):
    sft.main(['--input',str(inp),'--output',str(root/'sft'),'--max-examples','60'])
   manifest=json.loads((root/'sft'/'dataset_manifest.json').read_text(encoding='utf-8'))
   self.assertEqual(manifest['selected_examples'],60)
   assignments={}
   with (root/'sft'/'examples_metadata.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
     pid=r['procedure_id'];split=r['split']
     self.assertEqual(assignments.setdefault(pid,split),split)
   with patch.object(audit,'require_arrow',return_value=FakePQ()),patch.object(audit,'iter_parquet',side_effect=fake_iter):
    audit.main(['--input',str(inp),'--output',str(root/'audit')])
   summary=json.loads((root/'audit'/'audit_summary.json').read_text(encoding='utf-8'))
   self.assertEqual(summary['counts']['procedures'],50)
   self.assertEqual(summary['counts']['procedure_items'],50)
   self.assertEqual(summary['integrity']['items_without_procedure'],0)

if __name__=='__main__':unittest.main()
