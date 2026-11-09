"""Crop matched intensity/response pairs using identical pixel slices."""
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D, NoOverlapError
from .io_utils import image_pairs
from .background import prepare_background
import json


def crop_band(base_folder, band, ra, dec, crop_size, pairs=None, *,
              background_mode='archive', local_mesh_arcmin=6.0,
              local_galaxy_mask_factor=1.5):
    folder = Path(base_folder)
    output = folder / 'cropped' / band
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for intensity, response in (image_pairs(folder, band) if pairs is None else pairs):
        header = fits.getheader(intensity)
        if local_galaxy_mask_factor <= 1 or not np.isfinite(local_galaxy_mask_factor):
            raise ValueError('local_galaxy_mask_factor must be finite and > 1')
        radius_arcsec = float(crop_size.to_value('arcsec')) * 0.5 * local_galaxy_mask_factor
        data, residual, bg_report = prepare_background(
            intensity, response, mode=background_mode,
            center=SkyCoord(ra, dec, unit='deg'),
            galaxy_mask_radius_arcsec=radius_arcsec,
            mesh_arcmin=local_mesh_arcmin)
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
        out_header['BGSUB'] = (background_mode != 'none', 'GALEX background removed before stacking')
        out_header['BGMODE'] = (background_mode.upper(), 'Pre-stack GALEX background treatment')
        out_header['BGSTAGE'] = ('PRESTACK', 'Background correction processing stage')
        out_header.add_history('Input GALEX background mode: ' + background_mode)
        for key in ('EXPTIME', 'BUNIT'):
            if key in header:
                out_header[key] = header[key]
        ipath = output / intensity.name.replace('.fits', '_crop.fits')
        epath = output / response.name.replace('.fits', '_crop.fits')
        fits.writeto(ipath, cut.data, out_header, overwrite=True)
        eh = cut.wcs.to_header()
        eh['BUNIT'] = 's'
        fits.writeto(epath, ecut, eh, overwrite=True)
        if residual is not None:
            bgcut = np.full(cut.data.shape, np.nan)
            bgcut[cut.slices_cutout] = residual[cut.slices_original]
            bgpath = output / intensity.name.replace('-int.fits', '-localbg_crop.fits')
            fits.writeto(bgpath, bgcut, out_header, overwrite=True)
            bg_report['residual_cutout'] = str(bgpath)
        report_path = output / intensity.name.replace('-int.fits', '-background.json')
        report_path.write_text(json.dumps(bg_report, indent=2) + '\n')
        paths.append((ipath, epath))
    return paths

