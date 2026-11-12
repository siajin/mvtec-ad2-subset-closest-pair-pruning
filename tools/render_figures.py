"""Render repository figures from the released aggregate tables only.

No feature extraction, selection, inference, or ground-truth evaluation occurs.
Requires Matplotlib and NumPy in a separate plotting environment.
"""
from pathlib import Path
import csv
import os
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHODS=['SuperADD_style','SCPP','SCPP_PLUS_LOWTAIL','KMEANS_TWO_STAGE','KC100','LEARNED','PROJECTED']
LABELS=['SuperADD-style','SCPP','SCPP + low-tail','K-means\ntwo-stage','k-center','GT-learned','GT-projected']
PAPER='SuperADD paper-reported reference: pAUROC 83.93%, fixed F1 62.61%; separate conditions; geometry N/A.'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':16,'axes.labelsize':11,
    'xtick.labelsize':11,'ytick.labelsize':11,'figure.facecolor':'white','axes.facecolor':'white',
    'svg.fonttype':'none'})

def rows(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))

def matrix(values,title,unit,path,note,percent=False):
    fig,ax=plt.subplots(figsize=(11.6,9.2),layout=None)
    fig.subplots_adjust(left=.17,right=.86,bottom=.20,top=.84)
    masked=np.ma.masked_invalid(values)
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#e5e7eb')
    vmax=100 if percent else max(float(np.nanmax(values)),1e-12)
    img=ax.imshow(masked,cmap=cmap,vmin=0,vmax=vmax)
    ax.set_xticks(range(7),LABELS,rotation=35,ha='right')
    ax.set_yticks(range(7),LABELS)
    for i in range(7):
        for j in range(7):
            v=values[i,j]
            label='N/A' if not np.isfinite(v) else (f'{v:.2f}' if percent else f'{v:.3g}')
            ax.text(j,i,label,ha='center',va='center',fontsize=10,
                color='white' if np.isfinite(v) and v>.58*vmax else '#20252b')
    ax.set_xlabel('Target bank',labelpad=10);ax.xaxis.set_label_position('top')
    ax.set_ylabel('Source bank',labelpad=10)
    cb=fig.colorbar(img,ax=ax,fraction=.047,pad=.04);cb.set_label(unit)
    fig.suptitle(title,y=.96,fontweight='bold')
    fig.text(.5,.90,'SCPP = Subset-based Closest-Pair Pruning',ha='center',fontsize=12)
    fig.text(.5,.075,note,ha='center',fontsize=10)
    fig.text(.5,.035,PAPER,ha='center',fontsize=8.8)
    fig.savefig(ROOT/path,dpi=160)
    plt.close(fig)

def main():
    ids=rows('results/geometry/id_overlap_summary.csv')
    v=np.full((7,7),np.nan)
    for i,m in enumerate(METHODS):
        if m!='LEARNED':v[i,i]=100
    for r in ids:
        a=METHODS.index(r['method_a']);b=METHODS.index(r['method_b'])
        v[a,b]=100*float(r['mean_fraction_a_in_b']);v[b,a]=100*float(r['mean_fraction_b_in_a'])
    matrix(v,'Shared FIT identities','Shared IDs / source bank (%)','assets/fit_id_overlap.png',
        'Equal mean over 8 categories x 4 layers (32 banks); free prototypes have no FIT ID.',True)
    v=np.zeros((7,7))
    for r in rows('results/geometry/within_bank_spacing_cdf_summary.csv'):
        i=METHODS.index(r['method_a']);j=METHODS.index(r['method_b'])
        v[i,j]=v[j,i]=float(r['bank_median'])
    matrix(v,'Within-bank NN spacing: CDF difference','Median CDF supremum difference',
        'assets/within_bank_spacing_cdf.png','Median of 32 bank-level comparisons; distribution similarity is not positional identity.')
    directed=rows('results/geometry/cross_directional_1344.csv')
    for layer in [7,15,23,31]:
        v=np.zeros((7,7))
        for i,a in enumerate(METHODS):
            for j,b in enumerate(METHODS):
                if a==b:continue
                rr=[r for r in directed if int(r['layer'])==layer and r['source']==a and r['target']==b]
                assert len(rr)==8
                v[i,j]=mean(float(r['mean_RMS_NN']) for r in rr)
        matrix(v,f'Layer {layer}: cross-bank nearest-neighbor distance','Euclidean / sqrt(1280)',
            f'assets/cross_mean_L{layer}.png','Equal mean over 8 categories; full target bank; independent color scale per layer.')
    patch=rows('results/patchcore/per_category.csv')
    categories=[r['category'] for r in patch if r['method']=='SCPP']
    get=lambda m:np.array([100*float(next(r for r in patch if r['category']==c and r['method']==m)['AU_PRO_005']) for c in categories])
    proposed=get('SCPP');baseline=get('coreset');y=np.arange(8)
    fig,(a,b)=plt.subplots(1,2,figsize=(12.8,7.8),gridspec_kw={'width_ratios':[1.9,1]})
    fig.subplots_adjust(left=.12,right=.96,top=.81,bottom=.19,wspace=.47)
    a.barh(y-.19,proposed,height=.34,color='#245c89',label='SCPP')
    a.barh(y+.19,baseline,height=.34,color='#b8c7d2',edgecolor='#495663',label='coreset-100k')
    a.set_yticks(y,categories);a.invert_yaxis();a.set_xlim(0,75);a.set_xlabel('AU-PRO@0.05 (%)');a.legend(loc='lower right')
    delta=proposed-baseline
    b.barh(y,delta,color='#245c89',height=.56);b.axvline(0,color='#495663',linewidth=1)
    b.set_yticks(y,[]);b.invert_yaxis();b.set_xlim(-1.6,2.7);b.set_xlabel('SCPP - coreset (percentage points)')
    for i,x in enumerate(delta):b.text(x+(.06 if x>=0 else -.06),i,f'{x:+.3f}',va='center',ha='left' if x>=0 else 'right',fontsize=10)
    for ax in [a,b]:
        ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='x',color='#e5e7eb')
    fig.suptitle('PatchCore features: representative-selection comparison',y=.96,fontsize=17,fontweight='bold')
    fig.text(.5,.89,'Subset-based Closest-Pair Pruning vs. approximate greedy coreset',ha='center',fontsize=12)
    fig.text(.5,.09,'WRN50-2, 100k representatives, seed42; public good+bad; 4 wins / 4 losses.',ha='center',fontsize=10)
    fig.text(.5,.045,PAPER,ha='center',fontsize=8.8)
    fig.savefig(ROOT/'assets/patchcore_comparison.png',dpi=160);plt.close(fig)
    print('Rendered 7 figures from unchanged released tables.')

if __name__=='__main__':main()
