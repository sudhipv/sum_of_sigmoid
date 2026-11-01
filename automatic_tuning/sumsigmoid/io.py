"""Strict daily input and JSON handling; future counts never enter preparation."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np

def plain(value):
    """Encode undefined detector quantities as null, not nonstandard JSON NaN."""
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value

def save(path,value):
    Path(path).write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def day_number(value,origin):
    """Map numeric days or ISO calendar dates onto integer model days."""
    if isinstance(origin,str):
        return (dt.datetime.fromisoformat(str(value).replace('Z','+00:00')).date()
                -dt.date.fromisoformat(origin)).days
    result=float(value)-float(origin)
    if not np.isfinite(result) or abs(result-round(result))>1e-8:
        raise ValueError('Daily model times must be integers relative to time_origin')
    return int(round(result))

def load_training(config):
    """Read exact days0..cutoff; no interpolation, smoothing or rescaling of y.

    Only dates are parsed outside the training window. Their count fields are
    never converted or used. The detector receives a physical prefix copy.
    """
    data=config['data'];origin=config['time_origin'];cutoff=day_number(config['training_cutoff'],origin)
    if not 1<=cutoff<=159:
        raise ValueError('Frozen detector supports training cutoffs1..159; extension requires review')
    counts={}
    with Path(data['file']).open(newline='') as stream:
        reader=csv.reader(stream,delimiter=data.get('delimiter',','))
        headers=next(reader) if data.get('header',True) else None
        def column(key):
            value=data[key]
            if isinstance(value,int):return value
            if headers is None:raise ValueError('Named columns require header=true')
            return headers.index(value)
        dc,ic=column('date_column'),column('count_column')
        for row in reader:
            if not row:continue
            day=day_number(row[dc],origin)
            if day<0 or day>cutoff:continue
            if day in counts:raise ValueError(f'Duplicate training day{day}')
            value=float(row[ic])
            if not np.isfinite(value) or value<0:raise ValueError('Counts must be finite and nonnegative')
            counts[day]=value
    if set(counts)!=set(range(cutoff+1)):
        raise ValueError('Training days must be complete0..cutoff; missing days are not imputed')
    y=np.array([counts[k] for k in range(cutoff+1)],dtype=float)
    if not np.any(y>0):raise ValueError('An all-zero curve is outside the frozen detector domain')
    return y,cutoff
