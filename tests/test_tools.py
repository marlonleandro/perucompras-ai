import sys,unittest,json,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from validate_sft_dataset import validate
from benchmark_qwen3_base import metrics,parse_json_object
class ToolsTest(unittest.TestCase):
 def test_real_files(self):
  folder=Path('/mnt/data')
  if not (folder/'train.jsonl').exists():self.skipTest('Uploads no disponibles')
  summary,issues=validate(folder)
  self.assertEqual(sum(summary['counts'][f'{s}_rows'] for s in ['train','val','test']),200)
  self.assertEqual(summary['errors'],0,str(issues[:3]))
 def test_leak(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   row={'instruction':'Clasifica el objeto contractual de la contratacion peruana','input':json.dumps({'descripcion_del_item':'x','objeto_del_proceso':'y','objeto_contractual':'Bien'}),'output':json.dumps({'category':'Bien'})}
   for split in ['train','val','test']:(p/f'{split}.jsonl').write_text(json.dumps(row)+'\n')
   s,_=validate(p)
   self.assertGreater(s['errors'],0)
 def test_json_parser(self):
  self.assertEqual(parse_json_object('```json\n{"category":"Bien"}\n```'),{'category':'Bien'})
 def test_metrics(self):
  r=[dict(task='classification',expected={'category':'Bien'},parsed={'category':'Bien'},latency_seconds=1)]
  self.assertEqual(metrics(r)['tasks']['classification']['accuracy'],1)
if __name__=='__main__':unittest.main()
