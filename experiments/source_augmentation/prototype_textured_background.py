#!/usr/bin/env python3
"""Small appearance-only experiment on the existing two plant embeddings."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def standardized_noise(rng, shape, sigma):
    noise = gaussian_filter(rng.normal(size=shape), sigma)
    return (noise - noise.mean()) / max(noise.std(), 1e-8)


def render(raw, rng):
    # Generated signal, not RGB. Preserve its soft foreground occupancy.
    alpha = np.clip(raw + .5, 0, 1)
    slow = standardized_noise(rng, raw.shape, 8)
    texture = standardized_noise(rng, raw.shape, 1.2)
    fine = rng.normal(size=raw.shape)
    background = np.clip(.22 + .045 * slow + .035 * texture + .015 * fine, 0, 1)
    foreground = np.clip(.72 + .04 * slow + .025 * texture + .015 * fine, 0, 1)
    softened = gaussian_filter(alpha, .65)
    variants = np.stack([alpha, (1-alpha)*background + alpha*foreground,
                         (1-softened)*background + softened*foreground]).astype('float32')
    assert np.isfinite(variants).all() and variants.min() >= 0 and variants.max() <= 1
    assert np.array_equal(variants[1][alpha == 0], background.astype('float32')[alpha == 0])
    return variants, background.astype('float32')


def encode(a):
    return base64.b64encode(np.rint(np.clip(a, 0, 1)*255).astype('uint8').tobytes()).decode()


def preview(images, mask, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    indices = [int(np.argmax(mask.sum(axis=tuple(j for j in range(3) if j != axis)))) for axis in range(3)]
    fig, axes = plt.subplots(4, 3, figsize=(11, 13))
    for column, label in enumerate(['Generated input', 'Textured background', 'Texture + soft boundary']):
        axes[0, column].set_title(label)
        for axis in range(3):
            axes[axis, column].imshow(np.take(images[column], indices[axis], axis=axis).T,
                                     origin='lower', cmap='gray', vmin=0, vmax=1)
            axes[axis, column].set_ylabel(f'axis {axis}, slice {indices[axis]}')
        axes[3, column].imshow(images[column].max(axis=0).T, origin='lower', cmap='gray', vmin=0, vmax=1)
        axes[3, column].set_ylabel('MIP axis 0')
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title + ' — identical geometry, shared texture realization')
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def viewer(cases, path):
    html = '''<!doctype html><meta charset="utf-8"><title>Plant background prototype</title>
<style>body{font:15px system-ui;margin:20px;background:#eee}table{width:100%;text-align:center}canvas{width:27vw;max-width:360px;image-rendering:pixelated}td{padding:4px}label{margin-right:20px}</style>
<h2>Generated plant signal in a textured 3D background</h2>
<p>Same graph and foreground geometry in each column. Illustrative texture, not calibrated MRI. No training data changed.</p>
<select id="sample"></select> <label><input id="overlay" type="checkbox">Mask outline (orange), nearby nodes (red)</label>
<p id="sliders"></p><table><thead><tr><th>Generated input</th><th>Textured background</th><th>Texture + soft boundary</th></tr></thead><tbody id="panels"></tbody></table>
<p>Rows: slices normal to axes 0, 1, 2. Move sliders to inspect the background inside and outside the original slab. All panels use intensity range [0,1]. Softening changes appearance only; annotations are fixed. Background is generated throughout the cube, including the slab.</p><script>
const cases=PAYLOAD;
const pick=document.getElementById('sample'), overlay=document.getElementById('overlay');
const sliders=[], canvases=[]; let current;
Object.keys(cases).forEach(k=>pick.add(new Option(k,k)));
for(let a=0;a<3;a++){
 let label=document.createElement('label');label.textContent='Axis '+a+' ';
 let s=document.createElement('input');s.type='range';s.min=0;s.max=63;s.value=32;label.append(s);
 let number=document.createElement('span');label.append(number);s.oninput=()=>{number.textContent=s.value;draw()};sliders.push(s);document.getElementById('sliders').append(label);
 let row=document.createElement('tr');canvases[a]=[];
 for(let v=0;v<3;v++){let td=document.createElement('td'), c=document.createElement('canvas');c.width=c.height=256;td.append(c);row.append(td);canvases[a][v]=c;}document.getElementById('panels').append(row);
}
function decode(s){return Uint8Array.from(atob(s),c=>c.charCodeAt(0));}
function index(p){return (p[0]*64+p[1])*64+p[2];}
function draw(){if(!current)return;
 for(let a=0;a<3;a++){const dims=[0,1,2].filter(x=>x!==a), slice=+sliders[a].value;
 for(let v=0;v<3;v++){let ctx=canvases[a][v].getContext('2d'), im=ctx.createImageData(256,256);
 for(let y=0;y<256;y++)for(let x=0;x<256;x++){
 let p=[0,0,0];p[a]=slice;p[dims[0]]=Math.floor(x/4);p[dims[1]]=63-Math.floor(y/4);
 let i=index(p), value=current.images[v][i], k=(y*256+x)*4;
 let border=false;if(overlay.checked && current.mask[i]){for(let d of dims)for(let sign of [-1,1]){let q=p.slice();q[d]+=sign;if(q[d]<0||q[d]>63||!current.mask[index(q)])border=true;}}
 im.data[k]=border?255:value;im.data[k+1]=border?155:value;im.data[k+2]=border?0:value;im.data[k+3]=255;
 }ctx.putImageData(im,0,0);
 if(overlay.checked){ctx.fillStyle='red';for(let p of current.nodes)if(Math.abs(p[a]*64-slice)<1.5){ctx.beginPath();ctx.arc(p[dims[0]]*256,(63-p[dims[1]]*64)*4,3,0,7);ctx.fill();}}
 }}
}
function select(){let c=cases[pick.value];current={images:c.images.map(decode),mask:decode(c.mask),nodes:c.nodes};for(let a=0;a<3;a++){sliders[a].value=c.indices[a];sliders[a].nextSibling.textContent=c.indices[a];}draw();}
pick.onchange=select;overlay.onchange=draw;select();</script>'''
    path.write_text(html.replace('PAYLOAD', json.dumps(cases)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/source_nonplanar_20261016'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(364506); cases = {}; report = {}
    for name in ['sample_000001', 'sample_000002']:
        with np.load(a.input / (name+'.npz')) as data:
            for stage, label in [(0, 'planar'), (1, 'nonplanar')]:
                key = name+'_'+label; raw = data[f'{stage}_raw']; mask = data[f'{stage}_seg'] > 0
                images, background = render(raw, rng)
                nodes = data[f'{stage}_nodes']; guides = data[f'{stage}_guides']
                np.savez_compressed(a.output/(key+'.npz'), images=images, background=background,
                                    mask=mask, nodes=nodes, guides=guides, edges=data['edges'])
                preview(images, mask, key, a.output/(key+'.png'))
                indices = [int(np.argmax(mask.sum(axis=tuple(j for j in range(3) if j!=axis)))) for axis in range(3)]
                cases[key] = dict(images=[encode(v) for v in images], mask=encode(mask), nodes=nodes.tolist(), indices=indices)
                report[key] = dict(nodes=len(nodes), edges=len(data['edges']),
                    foreground_mean=[float(v[mask].mean()) for v in images],
                    background_mean=[float(v[~mask].mean()) for v in images],
                    background_std=[float(v[~mask].std()) for v in images],
                    raw_unique_values=len(np.unique(raw)), annotations_unchanged=True)
    viewer(cases, a.output/'textured-plants.html')
    (a.output/'report.json').write_text(json.dumps(dict(seed=364506, samples=report), indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
