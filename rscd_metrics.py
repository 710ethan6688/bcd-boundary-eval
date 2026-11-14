"""Core region/inner-band/nearest-contour metrics; explicit undefined values."""
import math
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

def validate_pair(pred,gt):
    if pred.dtype!=bool or gt.dtype!=bool or pred.ndim!=2 or pred.shape!=gt.shape:
        raise ValueError('Aligned 2D boolean masks are required; binarize labels explicitly')

def divide(n,d):return float(n/d) if d else None

def region_counts(pred,gt):
    validate_pair(pred,gt)
    return dict(tp=int((pred & gt).sum()),fp=int((pred & ~gt).sum()),
                fn=int((~pred & gt).sum()),tn=int((~pred & ~gt).sum()))

def region_scores(c):
    tp,fp,fn=c['tp'],c['fp'],c['fn']
    return dict(precision=divide(tp,tp+fp),recall=divide(tp,tp+fn),
                f1=divide(2*tp,2*tp+fp+fn),iou=divide(tp,tp+fp+fn))

def inner_band(mask,width=1):
    if mask.dtype!=bool or mask.ndim!=2:raise ValueError('Boolean 2D mask required')
    if isinstance(width,bool) or not isinstance(width,(int,np.integer)) or width<1:
        raise ValueError('Positive integer band width required')
    return mask & ~binary_erosion(mask,structure=np.ones((3,3),bool),iterations=int(width),border_value=0)

def exclude_frame(points,margin):
    result=points.copy()
    if margin:
        result[:margin,:]=False;result[-margin:,:]=False
        result[:,:margin]=False;result[:,-margin:]=False
    return result

def biou_counts(pred,gt,width=1,exclude_border=False):
    validate_pair(pred,gt)
    pb,gb=inner_band(pred,width),inner_band(gt,width)
    if exclude_border:pb,gb=exclude_frame(pb,width),exclude_frame(gb,width)
    return dict(intersection=int((pb & gb).sum()),union=int((pb | gb).sum()),
                pred_band=int(pb.sum()),gt_band=int(gb.sum()))

def bf1_counts(pred,gt,tolerance=1.,gsd=1.,exclude_border=False):
    validate_pair(pred,gt)
    if not math.isfinite(tolerance) or tolerance<0 or not math.isfinite(gsd) or gsd<=0:
        raise ValueError('Finite nonnegative tolerance and positive GSD required')
    pb,gb=inner_band(pred),inner_band(gt)
    if exclude_border:
        margin=math.ceil(tolerance/gsd)+1
        pb,gb=exclude_frame(pb,margin),exclude_frame(gb,margin)
    pm=int((distance_transform_edt(~gb,sampling=gsd)[pb]<=tolerance).sum()) if gb.any() else 0
    gm=int((distance_transform_edt(~pb,sampling=gsd)[gb]<=tolerance).sum()) if pb.any() else 0
    return dict(pred_matches=pm,gt_matches=gm,pred_boundary=int(pb.sum()),gt_boundary=int(gb.sum()))

def bf1_scores(c):
    p=divide(c['pred_matches'],c['pred_boundary']);r=divide(c['gt_matches'],c['gt_boundary'])
    if c['pred_boundary']==0 and c['gt_boundary']==0:f=None
    elif c['pred_boundary']==0 or c['gt_boundary']==0:f=0.
    elif p+r==0:f=0.
    else:f=2*p*r/(p+r)
    return dict(precision=p,recall=r,bf1=f)

def sum_counts(rows):
    if not rows:raise ValueError('No observations')
    keys=set(rows[0])
    if any(set(r)!=keys for r in rows):raise ValueError('Inconsistent count schema')
    return {key:sum(r[key] for r in rows) for key in keys}

def reassemble(patches,shape):
    """patches: (top,left,2D array), complete tiling required before boundary extraction."""
    if not patches:raise ValueError('No patches')
    result=np.zeros(shape,dtype=patches[0][2].dtype);covered=np.zeros(shape,bool)
    for top,left,patch in patches:
        if patch.ndim!=2 or patch.dtype!=result.dtype:raise ValueError('Inconsistent patch')
        h,w=patch.shape
        if top<0 or left<0 or top+h>shape[0] or left+w>shape[1]:raise ValueError('Out of bounds')
        if covered[top:top+h,left:left+w].any():raise ValueError('Overlapping patches')
        result[top:top+h,left:left+w]=patch;covered[top:top+h,left:left+w]=True
    if not covered.all():raise ValueError('Incomplete original-image coverage')
    return result
