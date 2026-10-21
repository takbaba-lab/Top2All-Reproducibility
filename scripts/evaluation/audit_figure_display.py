#!/usr/bin/env python3
"""Validate released figure wording and unchanged nontext SVG geometry."""
from pathlib import Path
import hashlib,json,re,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
def geometry(path):
    out=[]
    for e in ET.parse(path).getroot().iter():
        tag=e.tag.rsplit('}',1)[-1]
        if tag not in ('path','use','rect','circle','ellipse','line','polygon','polyline'):continue
        a={k:v for k,v in e.attrib.items() if k!='id'}
        for k,v in list(a.items()):
            a[k]=re.sub(r'#.*','#MARKER',v) if k.endswith('href') else re.sub(r'url\(#[^)]*\)','url(#CLIP)',v)
        out.append((tag,a))
    return out
def geometry_hash(path):
    return hashlib.sha256(json.dumps(geometry(path),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def validate_display():
    audit=json.loads((ROOT/'reproducibility/figure_display_audit.json').read_text())
    assert audit['status']=='PASS'
    checked=0
    for r in audit['figures']:
        path=ROOT/'figures'/(r['figure']+'.svg')
        assert geometry_hash(path)==r['geometry_sha256_before']==r['geometry_sha256_after'],r['figure']
        for e in ET.parse(path).getroot().iter():
            tag=e.tag.rsplit('}',1)[-1]
            if tag not in ('text','title','description'):continue
            s=''.join(e.itertext())
            # Upstream software's Creator metadata is a version, not project history.
            if s.startswith('Matplotlib v'):continue
            s=re.sub(r'true-variable|transcriptomic state','',s,flags=re.I)
            assert not re.search(r'\b(?:true|truth|state|canonical|V[1-4])\b',s,re.I),(r['figure'],s)
            checked+=1
    return dict(figures=len(audit['figures']),manuscript_text_nodes_checked=checked,nontext_geometry='unchanged')
if __name__=='__main__':print(json.dumps(validate_display(),indent=2))
