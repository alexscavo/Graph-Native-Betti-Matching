#!/usr/bin/env python3
"""CPU-only rescore of saved predictions; never changes checkpoints or legacy results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy
from metrics.vessel_graph import VesselGraph
from metrics.vessel_suite import Protocol,evaluate,aggregate

RUN_ROOT=Path('/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/scratch_extend300_20261012')
ARMS={'straight_final':'graph_control_final','bspline_detached_final':'detached_spline_final'}


def records(path):
    """Stream one JSON array record, avoiding the 260+ MB decoded-object list."""
    decoder=json.JSONDecoder();buffer='';finished=False;started=False
    with path.open() as stream:
        while True:
            if not buffer.strip() and not finished:
                chunk=stream.read(1024*1024);finished=not chunk;buffer+=chunk
            buffer=buffer.lstrip()
            if not started:
                if not buffer.startswith('['):raise ValueError('expected prediction array')
                buffer=buffer[1:];started=True
            buffer=buffer.lstrip(' \n\r\t,')
            if buffer.startswith(']'):return
            try:
                value,end=decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if finished:raise
                chunk=stream.read(1024*1024);finished=not chunk;buffer+=chunk;continue
            buffer=buffer[end:]
            yield value


def reference(path):
    with np.load(path) as cache:
        metric=cache['metric'].astype(float)
        nodes=cache['nodes']@metric.T;edges=cache['edges']
        paths=cache['full_paths']@metric.T;offsets=cache['path_offsets']
        curves=[paths[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    return VesselGraph.from_arrays(nodes,edges,curves),metric


def collect(output):
    if not all((output/a/'summary.json').exists() for a in ARMS):return
    summaries={a:json.loads((output/a/'summary.json').read_text()) for a in ARMS}
    first=next(iter(summaries.values()))
    for value in summaries.values():
        for key in ('sample_ids','dataset_root','protocol','reference_geometry_sha256'):
            assert value['provenance'][key]==first['provenance'][key],key
    columns=['model','samples','curve_f1_at_1mm','average_centerline_distance_mm','hd95_mm_mean',
             'topo_3d_f1','apls_3d','anchor_f1','junction_f1','branch_f1','betti0_error_mean',
             'betti1_error_mean','representation_nodes_mean','representation_internal_points_mean',
             'representation_geometry_connectivity_bytes_float32_int32_mean']
    rows=[{'model':a,**{k:v['metrics'].get(k) for k in columns[1:]}} for a,v in summaries.items()]
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    text='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    for row in rows:text+='| '+' | '.join(f'{row[k]:.5g}' if isinstance(row[k],float) else str(row[k]) for k in columns)+' |\n'
    (output/'comparison.md').write_text('# Vessel core v1 comparison\n\nSame saved final checkpoints and validation IDs; no new inference. '
        'TOPO/APLS are explicit 3D adaptations, not directly comparable with 2D road benchmark numbers. '
        'GT guides are uncertain; HD95 is the mean per-case value. Legacy metrics are preserved separately.\n\n'+text)
    (output/'comparison.json').write_text(json.dumps(dict(rows=rows,paired_protocol_verified=True,
        protocol=first['provenance']['protocol']),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--run-root',type=Path,default=RUN_ROOT)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--max-samples',type=int)
    args=parser.parse_args();protocol=Protocol()
    if args.max_samples is not None and args.max_samples<=0:parser.error('--max-samples must be positive')
    source=args.run_root/'full_validation'/ARMS[args.arm]
    prior=json.loads((source/'summary.json').read_text())
    ids=prior['provenance']['sample_ids'];assert len(ids)==len(set(ids))==1952
    if args.max_samples:ids=ids[:args.max_samples]
    output=args.output_root or args.run_root/('core_metrics_v1' if not args.max_samples else f'core_metrics_v1_smoke{args.max_samples}')
    destination=output/args.arm;destination.mkdir(parents=True,exist_ok=False)
    provenance=dict(source_model=ARMS[args.arm],source_checkpoint=prior['provenance']['checkpoint'],
        runtime_versions=dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__),
        predictions=str(source/'validation-000000/predictions.json'),dataset_root=prior['provenance']['dataset_root'],
        sample_ids=ids,protocol=protocol.metadata(),inference_performed=False,legacy_results_unchanged=True,
        reference='same uncertain full-path guides and topology as legacy common-v2',
        representation_bytes='minimal geometry/connectivity payload: float32 coordinates, int32 endpoints; shared endpoints; no logits or evaluator samples')
    (destination/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    rows=[];started=time.monotonic();reference_digest=hashlib.sha256();prediction_digest=hashlib.sha256()
    for i,record in enumerate(records(source/'validation-000000/predictions.json')):
        if i>=len(ids):break
        assert record['source_sample_id']==ids[i]
        gt,metric=reference(Path(provenance['dataset_root'])/'val/spline'/f'{ids[i]}.npz')
        predicted=VesselGraph.from_prediction(record,metric,curved=args.arm!='straight_final')
        for digest,graph in ((reference_digest,gt),(prediction_digest,predicted)):
            digest.update(ids[i].encode())
            for value in (graph.nodes_mm,graph.edges,*graph.curves_mm):
                value=np.asarray(value)
                digest.update(str(value.shape).encode());digest.update(value.tobytes())
        row=dict(sample_id=ids[i],**evaluate(predicted,gt,protocol));rows.append(row)
        if (i+1)%50==0:print(json.dumps(dict(arm=args.arm,completed=i+1,elapsed_seconds=time.monotonic()-started)),flush=True)
    assert len(rows)==len(ids)
    with (destination/'per-patch.json').open('w') as stream:json.dump(rows,stream,indent=2,allow_nan=False)
    with (destination/'per-patch.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    metrics=aggregate(rows,protocol)
    provenance.update(reference_geometry_sha256=reference_digest.hexdigest(),
                      evaluated_prediction_geometry_sha256=prediction_digest.hexdigest())
    (destination/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    summary=dict(provenance=provenance,metrics=metrics,elapsed_seconds=time.monotonic()-started,
        legacy_native=prior['native'],legacy_common=prior['common'])
    pending=destination/'summary.pending.json'
    pending.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    pending.replace(destination/'summary.json')
    print(json.dumps(dict(arm=args.arm,elapsed_seconds=summary['elapsed_seconds'],metrics=metrics)),flush=True)
    collect(output)


if __name__=='__main__':main()
