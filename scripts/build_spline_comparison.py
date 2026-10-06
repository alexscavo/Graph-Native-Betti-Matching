#!/usr/bin/env python3
"""Bundle completed learning runs into one offline comparison viewer."""
import argparse
import json
from pathlib import Path
from plotly.offline import get_plotlyjs
from vessel_mesh import mask_mesh


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--targets',type=Path,default=Path('experiments/spline_learning/targets_k8'))
    args=p.parse_args()
    records={path.parent.name:json.loads(path.read_text()) for path in sorted(args.run_root.glob('v*/predictions.json'))}
    assert records
    meshes={}
    for name,rows in records.items():
        for row in rows:
            sample=row['sample']
            if sample not in meshes:meshes[sample]=mask_mesh(args.targets/'train/seg'/f'{sample}_seg.nii.gz')
            row['mesh']=meshes[sample]
        template=(Path(__file__).parent/'spline_predictions.html').read_text()
        (args.run_root/name/'predictions.html').write_text(template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(rows,separators=(',',':'))))
    # Embed each patch mesh once in the combined viewer, shared by the runs.
    for rows in records.values():
        for row in rows:row.pop('mesh')
    template=(Path(__file__).parent/'spline_predictions.html').read_text()
    template=template.replace('Spline learning · four-patch diagnostic','Spline learning · experiment comparison')
    template=template.replace('<label>Patch ', '<label>Run <select id="run"></select></label><label>Patch ')
    template=template.replace('<option value="decoded">','<option value="decoded" selected>')
    template=template.replace('const DATA=__DATA__;', '''const ALL_DATA=__DATA__;
const MESHES=__MESHES__;
Object.values(ALL_DATA).forEach(rows=>rows.forEach(r=>r.mesh=MESHES[r.sample]));
const run=document.getElementById('run');
const labels={v0_pretrained:'V0 · controls · fine-tuned',v0_scratch:'V0 · controls · scratch',v1_pretrained:'V1 · sampled curve · fine-tuned'};
Object.keys(ALL_DATA).forEach(k=>run.add(new Option(labels[k]||k,k)));
let DATA=ALL_DATA[run.value];''')
    template=template.replace('patch.onchange=render;mode.onchange=render;render();',
                              'run.onchange=()=>{DATA=ALL_DATA[run.value];render();};patch.onchange=render;mode.onchange=render;render();')
    page=template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(records,separators=(',',':'))).replace('__MESHES__',json.dumps(meshes,separators=(',',':')))
    (args.run_root/'comparison.html').write_text(page)
    print('Saved',args.run_root/'comparison.html')


if __name__=='__main__':main()
