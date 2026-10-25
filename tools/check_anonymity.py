#!/usr/bin/env python3
"""Inspect release files, archive member names, NumPy headers and SHA256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TEXT_SUFFIXES={'.py','.md','.txt','.json','.jsonl','.csv','.yml','.yaml','.toml','.sh'}


def check(extra=()):
    findings=[];checked=0;arrays=0
    patterns={
        'personal_absolute_path':re.compile('/'+r'(?:home|Users)/[A-Za-z0-9_.-]+|[A-Za-z]:\\'+'Users'+r'\\',re.I),
        'email':re.compile(r'[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
        'github_token':re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})'),
        'private_key':re.compile('BEGIN '+r'(?:RSA |OPENSSH |EC )?PRIVATE KEY'),
    }
    paths=[]
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if any(x in ('.git','__pycache__','.cache','.venv','outputs') for x in rel.parts):continue
        if p.is_symlink():findings.append(dict(file=str(rel),kind='symlink'));continue
        if p.name in ('.env','hosts.yml','id_rsa','id_ed25519'):
            findings.append(dict(file=str(rel),kind='private_or_vcs_file'))
        if not p.is_file():continue
        checked+=1;paths.append(p)
        if p.suffix in TEXT_SUFFIXES or p.name=='.gitignore':
            text=p.read_text()
            for name,pattern in patterns.items():
                if pattern.search(text):findings.append(dict(file=str(rel),kind=name))
            for term in extra:
                if term.lower() in text.lower():findings.append(dict(file=str(rel),kind='user_supplied_identity_token'))
        elif p.suffix=='.npz':
            with zipfile.ZipFile(p) as z:
                for name in z.namelist():
                    if any(pattern.search(name) for pattern in patterns.values()):
                        findings.append(dict(file=str(rel),kind='archive_member_identity'))
            with np.load(p,allow_pickle=False) as z:
                for name in z.files:
                    arr=z[name];arrays+=1
                    if arr.dtype.hasobject or arr.dtype.kind in ('U','S'):
                        findings.append(dict(file=str(rel),kind='unexpected_nonnumeric_array'))
    manifest=ROOT/'MANIFEST.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())['files']
        actual={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p!=manifest}
        if actual!=saved:findings.append(dict(file='MANIFEST.json',kind='manifest_mismatch'))
    return dict(passed=not findings,files_checked=checked,numeric_arrays_checked=arrays,
        excludes_runtime_caches=True,git_history_scanned=False,findings=findings)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--forbid',action='append',default=[],help='Additional author/account/institution token; do not pass secrets')
    a=ap.parse_args();result=check(a.forbid)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
