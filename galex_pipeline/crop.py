"""Crop matched intensity/response pairs using identical pixel slices."""
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D, NoOverlapError
from .io_utils import image_pairs


def crop_band(base_folder, band, ra, dec, crop_size):
    folder = Path(base_folder)
    output = folder / 'cropped' / band
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for intensity, response in image_pairs(folder, band):
        data, header = fits.getdata(intensity, header=True)
        exposure, eheader = fits.getdata(response, header=True)
        wcs, ewcs = WCS(header).celestial, WCS(eheader).celestial
        if data.ndim != 2 or data.shape != exposure.shape:
            raise ValueError(f"Intensity/rrhr shapes differ: {intensity.name}")
        # Check the two grids in pixel coordinates before sharing a cutout slice.
        yy, xx = np.meshgrid(np.linspace(0, data.shape[0]-1, 3),
                             np.linspace(0, data.shape[1]-1, 3), indexing='ij')
        ex, ey = ewcs.world_to_pixel(wcs.pixel_to_world(xx, yy))
        if not (np.allclose(ex, xx, atol=1e-3, rtol=0) and np.allclose(ey, yy, atol=1e-3, rtol=0)):
            raise ValueError(f"Intensity/rrhr WCS mismatch: {intensity.name}")
        position = SkyCoord(ra, dec, unit='deg')
        try:
            cut = Cutout2D(data.astype(float), position, crop_size, wcs=wcs,
                           mode='partial', fill_value=np.nan)
        except NoOverlapError:
            print(f"No overlap: {intensity.name}")
            continue
        ecut = np.zeros(cut.data.shape, dtype=float)
        ecut[cut.slices_cutout] = exposure[cut.slices_original]
        out_header = cut.wcs.to_header()
        for key in ('EXPTIME', 'BUNIT'):
            if key in header:
                out_header[key] = header[key]
        ipath = output / intensity.name.replace('.fits', '_crop.fits')
        epath = output / response.name.replace('.fits', '_crop.fits')
        fits.writeto(ipath, cut.data, out_header, overwrite=True)
        eh = cut.wcs.to_header()
        eh['BUNIT'] = 's'
        fits.writeto(epath, ecut, eh, overwrite=True)
        paths.append((ipath, epath))
    return paths
