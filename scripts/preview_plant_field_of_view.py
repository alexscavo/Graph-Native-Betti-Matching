#!/usr/bin/env python3
"""Compare native and wider original plant crops at fixed 64px display input."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
root=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/plants_3d2cut')
m=json.loads((root/'prova/metadata.json').read_text());im=Image.open(m['source_image']).convert('RGB');seg=Image.open(m['source_segmentation']).convert('L')
f=json.loads(Path(m['source_annotation']).read_text())['VineImage'][0]['VineFeature'][0]
xy=np.asarray([v['FeatureCoordinates'][:2] for v in f]);centre=np.asarray(m['junction_original_xy']);a.output.mkdir(parents=True,exist_ok=False)
sizes=[64,256,708,1024];canvas=Image.new('RGB',(4*320,710),'white');draw=ImageDraw.Draw(canvas);rows=[]
for i,size in enumerate(sizes):
    start=np.rint(centre-size/2).astype(int);box=(*start,*(start+size));crop=im.crop(box);mask=seg.crop(box)
    small=crop.resize((64,64),Image.Resampling.BOX)
    # Occupancy keeps thin mask branches visible; this is a FOV illustration,
    # not the smoothed-mask input of the actual training loader.
    binary=Image.fromarray((np.asarray(mask)>0).astype('uint8')*255)
    mask64=Image.fromarray((np.asarray(binary.resize((64,64),Image.Resampling.BOX))>0).astype('uint8')*255)
    crop.save(a.output/f'crop_{size}px.png');small.save(a.output/f'input_{size}px_to64.png');mask64.save(a.output/f'mask_{size}px_to64.png')
    inside=np.all((xy>=start)&(xy<start+size),axis=1);n=int(inside.sum())
    draw.text((i*320+8,8),f'{size} x {size} source -> 64 x 64',fill='black');draw.text((i*320+8,28),f'{n} annotated points (includes degree-2)',fill='black')
    canvas.paste(small.resize((304,304),Image.Resampling.NEAREST),(i*320+8,55));canvas.paste(mask64.convert('RGB').resize((304,304),Image.Resampling.NEAREST),(i*320+8,385))
    rows.append(dict(original_crop_size=size,start_xy=start.tolist(),annotated_points=n))
canvas.save(a.output/'comparison.png');(a.output/'metadata.json').write_text(json.dumps(dict(source=m,examples=rows,note='Illustrates field of view. Actual training uses mask-derived input, 128px intermediate and 54px inner projection plus padding; source FOV about708px at downscale6.'),indent=2)+'\n')
print(a.output/'comparison.png')
