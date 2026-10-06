#!/usr/bin/env python3
"""Import declared dependencies in isolated processes and record active paths/versions.

Run from the repository after sourcing env.sh and stage_spline_runtime.sh.
"""
import json,subprocess,sys
from pathlib import Path

def main():
    aliases={'markdown-it-py':'markdown_it','absl-py':'absl','fonttools':'fontTools','grpcio':'grpc','hf-xet':'hf_xet','huggingface-hub':'huggingface_hub','importlib-resources':'importlib_resources','POT':'ot','pillow':'PIL','protobuf':'google.protobuf','python-dateutil':'dateutil','pytorch-ignite':'ignite','pyyaml':'yaml','scikit-image':'skimage','scikit-learn':'sklearn','simpleitk':'SimpleITK','typing-extensions':'typing_extensions'}
    requirements=[]
    for line in Path('requirements/jean-zay.txt').read_text().splitlines():
     if line and not line.startswith('#'):
      name,version=line.split('==');requirements.append((name,aliases.get(name,name.replace('-','_')),version))
    requirements += [(x,x,None) for x in ('torch','torchvision','torchaudio','wandb','MultiScaleDeformableAttention3D')]
    results=[];target=Path('experiments/spline_learning/metadata/library-availability.json')
    for name,module,pin in requirements:
     code=('import torch; ' if module=='MultiScaleDeformableAttention3D' else '')+'import importlib,json; m=importlib.import_module('+repr(module)+'); print("LIBRARY_RESULT="+json.dumps({"active_version":str(getattr(m,"__version__","")),"path":getattr(m,"__file__",None)}))'
     try:
      p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=60)
      data=next((json.loads(l.split('=',1)[1]) for l in p.stdout.splitlines() if l.startswith('LIBRARY_RESULT=')),None)
      record={'requirement':name,'module':module,'pinned_version':pin,'import':'OK' if p.returncode==0 and data is not None else 'FAILED'}
      if data:record.update(data)
      if record['import']=='FAILED':record['error']=(p.stderr or p.stdout)[-1600:]
     except subprocess.TimeoutExpired:record={'requirement':name,'module':module,'import':'FAILED','error':'Import timed out after 60 seconds'}
     results.append(record);target.write_text(json.dumps(results,indent=2)+'\n')
     print(record['import'],name,record.get('active_version',''),flush=True)
    failures=[r for r in results if r['import']=='FAILED'];print('Checked',len(results),'libraries; failed',len(failures),flush=True);print(json.dumps(failures,indent=2),flush=True)
    if failures:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
