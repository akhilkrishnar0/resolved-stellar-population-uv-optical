"""Coadd with spatial GALEX relative-response weights, not header exposure."""
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from reproject import reproject_interp


def stack_band(crop_folder, band, pairs=None):
    folder = Path(crop_folder)
    if pairs is None:
        pairs = [(p, p.with_name(p.name.replace('-int_crop', '-rrhr_crop')))
                 for p in sorted((folder / band).glob('*-int_crop.fits'))]
    if not pairs:
        print(f"No overlapping {band} image pairs")
        return None
    ref = fits.getheader(pairs[0][0])
    shape = fits.getdata(pairs[0][0]).shape
    numerator, weight = np.zeros(shape), np.zeros(shape)
    coverage = np.zeros(shape, dtype=np.int32)
    nominal = []
    ref_area = proj_plane_pixel_area(WCS(ref).celestial)
    for ipath, epath in pairs:
        data, header = fits.getdata(ipath, header=True)
        response = fits.getdata(epath)
        if data.shape != response.shape:
            raise ValueError(f"Mismatched cropped pair: {ipath}")
        area = proj_plane_pixel_area(WCS(header).celestial)
        if not np.isclose(area, ref_area, rtol=1e-5, atol=0):
            raise ValueError("Input pixel areas differ; counts/s/pixel requires explicit area conversion")
        valid = np.isfinite(data) & np.isfinite(response) & (response > 0)
        effective = np.where(valid, response, 0.)
        counts = np.zeros(data.shape)
        np.multiply(data, effective, out=counts, where=valid)
        n, footprint = reproject_interp((counts, header), ref, shape_out=shape)
        e, _ = reproject_interp((effective, header), ref, shape_out=shape)
        good = (footprint > 0) & np.isfinite(n) & np.isfinite(e) & (e > 0)
        numerator[good] += n[good]
        weight[good] += e[good]
        coverage[good] += 1
        if 'EXPTIME' in header and np.isfinite(header['EXPTIME']):
            nominal.append(float(header['EXPTIME']))
    if not np.any(weight > 0):
        raise ValueError(f'No positive valid response in {band}; no stack written')
    out = np.full(shape, np.nan)
    np.divide(numerator, weight, out=out, where=weight > 0)
    header = WCS(ref).celestial.to_header()
    header['NCOMBINE'] = len(pairs)
    header['WGHTTYPE'] = 'RRHR'
    header['PIPEVER'] = '0.2.0'
    if len(nominal) == len(pairs):
        header['TOTEXPT'] = (sum(nominal), 'Sum of nominal EXPTIME; NOT per-pixel exposure')
    header.add_history('Stack = sum(reproject(int * rrhr)) / sum(reproject(rrhr)).')
    header.add_history('Invalid input pixels have zero weight; uncovered output is NaN.')
    for ipath, _ in pairs:
        header.add_history('Input: ' + Path(ipath).name)
    outputs = {}
    for suffix, array, unit in (('stacked', out, 'count/s/pixel'),
                                 ('rrhr', weight, 's'), ('coverage', coverage, '1')):
        h = header.copy()
        h['BUNIT'] = unit
        path = folder / f'{band}_{suffix}.fits'
        fits.writeto(path, array, h, overwrite=True)
        outputs[suffix] = path
    return outputs
