"""Pack reviewed smoke/abandoned runs, verify bytes, then leave restore pointers."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

here = Path(__file__).resolve().parent
plan = json.loads((here/'archive-plan.json').read_text())
root = Path(plan['root'])
archive = root/'_archive'/'20261016-smoke-and-superseded-evidence.tar.gz'
assert not archive.exists()
assert not subprocess.check_output(['squeue', '-h', '-u', 'upz25mj', '-o', '%i'], text=True).strip()
directories = [Path(p) for p in plan['directories']]
before = {}
for directory in directories:
    assert directory.resolve().is_relative_to(root) and not directory.is_symlink()
    for base, dirs, names in os.walk(directory, followlinks=False):
        for name in dirs + names:
            p = Path(base)/name
            assert p.suffix not in {'.pt', '.pth', '.ckpt'}, p
            s = p.lstat()
            before[str(p.relative_to(root))] = (s.st_size, s.st_mtime_ns, s.st_mode)
archive.parent.mkdir(exist_ok=True)
with tarfile.open(archive, 'x:gz', compresslevel=1, dereference=False) as tar:
    for directory in directories:
        tar.add(directory, arcname=str(directory.relative_to(root)))
print('Archive created; verifying every archived file against its original.', flush=True)

def digest(stream):
    h = hashlib.sha256()
    while chunk := stream.read(1024*1024):
        h.update(chunk)
    return h.hexdigest()

files = 0
logical_bytes = 0
verified = {}
with tarfile.open(archive, 'r:gz') as tar:
    for member in tar:
        path = root/member.name
        s = path.lstat()
        if member.name in before:
            assert before[member.name] == (s.st_size, s.st_mtime_ns, s.st_mode), path
        if member.isfile():
            with path.open('rb') as original, tar.extractfile(member) as packed:
                sha = digest(original)
                assert sha == digest(packed), path
            files += 1; logical_bytes += s.st_size
            verified[member.name] = dict(bytes=s.st_size, sha256=sha)
        elif member.issym():
            assert os.readlink(path) == member.linkname
            verified[member.name] = dict(symlink=member.linkname)
        else:
            assert member.isdir(), 'Unexpected archive entry type'
with archive.open('rb') as stream:
    archive_sha = digest(stream)
(here/'archive-verification.json').write_text(json.dumps(dict(archive=str(archive), sha256=archive_sha,
    compressed_bytes=archive.stat().st_size, original_regular_bytes=logical_bytes,
    regular_files=files, verified=verified), indent=2)+'\n')
for directory in directories:
    shutil.rmtree(directory)
    directory.mkdir()
    (directory/'ARCHIVED.md').write_text(
        '# Archived startup / superseded evidence\n\n'
        f'Byte-verified archive: `{archive}`.\n\n'
        f'Restore this run: `tar -xzf {archive} -C {root} {directory.relative_to(root)}`\n\n'
        'Disposable smoke/cancelled checkpoints and random initial predictions were removed separately; '
        'the archive preserves the remaining logs, configurations, metrics and evidence. '
        'See repository docs/research/storage/cleanup-20261016.md.\n')
print(f'Archived {len(directories)} run directories; {files} regular files preserved; {archive.stat().st_size/2**20:.1f} MiB archive.', flush=True)
