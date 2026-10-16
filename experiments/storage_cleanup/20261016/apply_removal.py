"""Execute this reviewed, fixed cleanup manifest; refuses changed files."""
import argparse
import json
from pathlib import Path
import stat
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--apply', action='store_true')
a = p.parse_args()
here = Path(__file__).resolve().parent
plan = json.loads((here/'removal-plan.json').read_text())
root = Path(plan['root']).resolve()
for item in plan['items']:
    path = Path(item['path'])
    assert path.resolve().is_relative_to(root) and not path.is_symlink()
    s = path.stat()
    assert stat.S_ISREG(s.st_mode)
    assert (s.st_size, s.st_mtime_ns, s.st_ino, s.st_dev, s.st_nlink) == (
        item['bytes'], item['mtime_ns'], item['inode'], item['device'], item['nlink']), path
    assert s.st_nlink == 1, 'Shared hard link requires separate review'
print(len(plan['items']), 'verified files;', sum(i['bytes'] for i in plan['items'])/2**30, 'GiB logical', flush=True)
if a.apply:
    jobs = subprocess.check_output(['squeue', '-h', '-u', 'upz25mj', '-o', '%i'], text=True)
    assert not jobs.strip(), 'Jobs are active; re-audit dependencies before cleanup'
    with (here/'removed.jsonl').open('x') as log:
        for item in plan['items']:
            path = Path(item['path'])
            blocks = path.stat().st_blocks * 512
            path.unlink()
            log.write(json.dumps(dict(**item, allocated_bytes=blocks, removed_unix=time.time()))+'\n')
            log.flush()
    print('Removed only manifest entries; remaining run artifacts preserved.', flush=True)
