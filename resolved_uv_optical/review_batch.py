"""Nonfatal DECaLS batch review with separate readable/invalid PDF reports."""
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import simple_norm
from photutils.aperture import SkyCircularAperture


def review_batch(pipe, catalog, *, size=3000, download="missing", reject=None, circle_arcsec=60):
    table = catalog.to_pandas() if hasattr(catalog,"to_pandas") else pd.DataFrame(catalog).copy()
    required = ["galaxy","ra_deg","dec_deg","distance","size"]
    if table.empty or not set(required).issubset(table):
        raise ValueError(f"Provide a nonempty catalogue with {required}")
    if download not in (True,False,"missing"):
        raise ValueError("download must be True, False or 'missing'")
    if not np.isfinite(circle_arcsec) or circle_arcsec<=0:
        raise ValueError("circle_arcsec must be positive")
    if table.galaxy.isna().any():
        raise ValueError("Missing galaxy name")
    table['galaxy'] = table.galaxy.astype(str).str.replace(' ','',regex=False)
    if table.galaxy.duplicated().any() or any(g in ('','.','..') or '/' in g or '\\' in g for g in table.galaxy):
        raise ValueError("Invalid or duplicate galaxy names")
    reject = set(reject or [])
    if reject-set(table.galaxy):
        raise ValueError("Rejected names not in catalogue")
    out = pipe.base_dir/'decals_review'
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    counts={'good':0,'invalid':0}
    with PdfPages(out/'good.pdf') as good_pdf, PdfPages(out/'invalid.pdf') as invalid_pdf:
        for i in range(len(table)):
            one=table.iloc[[i]]
            row=one.iloc[0]
            name=row.galaxy
            reason=[]
            try:
                pipe.review_decals(one,size=size,download=download)
            except Exception as exc:
                reason.append(f"Review/download failed: {type(exc).__name__}: {exc}")
            if name in reject:
                reason.append('Manually rejected')
            path=pipe.decals.download_dir/'fits'/f'{name}.fits'
            data=wcs=center=None
            try:
                center=SkyCoord(float(row.ra_deg),float(row.dec_deg),unit='deg')
                with fits.open(path) as hdus:
                    data=np.array(hdus[0].data,dtype=float)
                    wcs=WCS(hdus[0].header).celestial
                if data.ndim!=3 or data.shape[0]!=3:
                    raise ValueError('Expected g/r/z primary cube')
                x,y=wcs.world_to_pixel(center)
                ny,nx=data.shape[1:]
                if not np.isfinite([x,y]).all() or not 0<=x<nx or not 0<=y<ny:
                    reason.append('Catalogue centre outside field')
                elif not np.isfinite(data[:,int(y),int(x)]).all():
                    reason.append('Nonfinite centre pixel in at least one band')
                if any(not np.isfinite(plane).any() for plane in data):
                    reason.append('No finite pixels in at least one band')
            except Exception as exc:
                data=None
                reason.append(f"Science FITS unavailable/invalid: {exc}")
            status='invalid' if reason else 'good'
            fig,axes=plt.subplots(1,3,figsize=(12,4.8))
            try:
                for j,(ax,band) in enumerate(zip(axes,('g','r','z'))):
                    ax.set_title(band)
                    if data is None:
                        ax.text(.5,.5,'No usable science FITS',ha='center',transform=ax.transAxes)
                        ax.set_axis_off()
                        continue
                    finite=np.isfinite(data[j])
                    if finite.any():
                        ax.imshow(data[j],origin='lower',cmap='gray',norm=simple_norm(data[j][finite],'asinh',percent=99.5))
                    x,y=wcs.world_to_pixel(center)
                    ny,nx=data.shape[1:]
                    if np.isfinite([x,y]).all() and 0<=x<nx and 0<=y<ny:
                        SkyCircularAperture(center,r=circle_arcsec*u.arcsec).to_pixel(wcs).plot(ax=ax,color='cyan',lw=1.2)
                        ax.plot(x,y,'r+',ms=8)
                    else:
                        ax.text(.02,.95,'Centre outside field',va='top',color='red',transform=ax.transAxes)
                    ax.set(xlim=(-.5,nx-.5),ylim=(-.5,ny-.5))
                fig.suptitle(f'{name} | {status} | RA={row.ra_deg}, Dec={row.dec_deg} deg')
                note='; '.join(reason) if reason else 'Readable g/r/z cube with finite centre pixels; visual approval still required.'
                fig.text(.02,.02,textwrap.fill(note[:800],150),fontsize=7,va='bottom')
                fig.tight_layout(rect=(0,.22,1,.93))
                (invalid_pdf if reason else good_pdf).savefig(fig)
            finally:
                plt.close(fig)
            counts[status]+=1
            rows.append(dict(galaxy=name,status=status,reason='; '.join(reason),science_fits=str(path)))
            pd.DataFrame(rows).to_csv(out/'review_status.csv',index=False)
        for status,pdf in [('good',good_pdf),('invalid',invalid_pdf)]:
            if not counts[status]:
                fig=plt.figure(figsize=(8,3))
                fig.text(.1,.5,f'No galaxies classified as {status} in this run.')
                pdf.savefig(fig)
                plt.close(fig)
    return pd.DataFrame(rows)
