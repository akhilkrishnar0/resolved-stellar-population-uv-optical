"""Pre-stack GALEX background treatment on each full-resolution observation.

Never alter the downloaded archive products. Archive-subtracted images provide the
baseline; an additional spatially varying correction is optional and must be inspected.
"""
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats


def _same_grid(header_a, header_b, shape):
    a, b = WCS(header_a).celestial, WCS(header_b).celestial
    ny, nx = shape
    yy, xx = np.meshgrid(np.linspace(0, ny - 1, 3),
                         np.linspace(0, nx - 1, 3), indexing="ij")
    bx, by = b.world_to_pixel(a.pixel_to_world(xx, yy))
    return (np.all(np.isfinite(bx)) and np.all(np.isfinite(by))
            and np.allclose(bx, xx, atol=1e-3, rtol=0)
            and np.allclose(by, yy, atol=1e-3, rtol=0))


def prepare_background(intensity_path, response_path, *, mode, center,
                       galaxy_mask_radius_arcsec, mesh_arcmin=6.0):
    """Return full-field corrected intensity, optional residual model and QA.

    mode='none' retains original intensity; 'archive' uses matched intbgsub;
    'local' removes a conservative masked Background2D residual from intbgsub.
    Never derive an additional local correction from the original unsubtracted int.
    """
    if mode not in ("none", "archive", "local"):
        raise ValueError("background_mode must be 'none', 'archive' or 'local'")
    intensity_path, response_path = Path(intensity_path), Path(response_path)
    intensity, ih = fits.getdata(intensity_path, header=True)
    response, rh = fits.getdata(response_path, header=True)
    if intensity.ndim != 2 or intensity.shape != response.shape:
        raise ValueError(f"Intensity/rrhr shapes differ: {intensity_path}")
    if not _same_grid(ih, rh, intensity.shape):
        raise ValueError(f"Intensity/rrhr WCS mismatch: {intensity_path}")
    valid = np.isfinite(intensity) & np.isfinite(response) & (response > 0)
    details = {"mode": mode, "input": str(intensity_path),
               "response": str(response_path)}
    residual = None
    if mode == "none":
        corrected = np.asarray(intensity, dtype=float)
    else:
        archive_path = intensity_path.parent.parent / "intbgsub" / intensity_path.name.replace("-int.fits", "-intbgsub.fits")
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing GALEX archive background-subtracted image: {archive_path}")
        corrected, ah = fits.getdata(archive_path, header=True)
        if corrected.shape != intensity.shape or not _same_grid(ih, ah, intensity.shape):
            raise ValueError(f"Archive intbgsub is not on the intensity WCS grid: {archive_path}")
        corrected = np.asarray(corrected, dtype=float)
        valid &= np.isfinite(corrected)
        details["archive_input"] = str(archive_path)
        if mode == "local":
            if not np.isfinite(mesh_arcmin) or mesh_arcmin <= 0:
                raise ValueError("mesh_arcmin must be positive and finite")
            if not np.isfinite(galaxy_mask_radius_arcsec) or galaxy_mask_radius_arcsec <= 0:
                raise ValueError("galaxy_mask_radius_arcsec must be positive and finite")
            from astropy.stats import SigmaClip
            from photutils.background import Background2D, MedianBackground
            from scipy.ndimage import binary_dilation
            from astropy.wcs.utils import proj_plane_pixel_scales
            scale = float(np.mean(np.abs(proj_plane_pixel_scales(WCS(ih).celestial)))) * 3600
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid GALEX pixel scale: {intensity_path}")
            pixels = max(8, int(round(mesh_arcmin * 60 / scale)))
            pixels = min(pixels, min(corrected.shape) // 2)
            if pixels < 8:
                raise ValueError(f"Image too small for local background grid: {intensity_path}")
            x0, y0 = WCS(ih).celestial.world_to_pixel(SkyCoord(center))
            yy, xx = np.ogrid[:corrected.shape[0], :corrected.shape[1]]
            galaxy = (xx-x0)**2 + (yy-y0)**2 <= (galaxy_mask_radius_arcsec / scale)**2
            sky = valid & ~galaxy
            if sky.sum() < 500 or sky.sum() < 0.2 * valid.sum():
                raise ValueError(f"Insufficient galaxy-free coverage for local GALEX background: {intensity_path}")
            _, med, std = sigma_clipped_stats(corrected[sky], sigma=3, maxiters=5)
            if not np.isfinite(std) or std <= 0:
                raise ValueError(f"Cannot estimate local background noise: {intensity_path}")
            source_mask = (corrected > med + 3 * std) & valid
            source_mask = binary_dilation(source_mask, iterations=4)
            bad = galaxy | source_mask | ~valid
            ny, nx = corrected.shape
            usable = 0
            for y in range(0, ny, pixels):
                for x in range(0, nx, pixels):
                    box = bad[y:min(y+pixels,ny), x:min(x+pixels,nx)]
                    if box.size >= 64 and np.mean(box) <= 0.5:
                        usable += 1
            if usable < 4 or np.mean(~bad) < 0.15:
                raise ValueError(f"Too few unmasked background meshes ({usable}) for {intensity_path}")
            model = Background2D(corrected, box_size=(pixels, pixels),
                                 mask=galaxy | source_mask, coverage_mask=~valid,
                                 sigma_clip=SigmaClip(sigma=3, maxiters=5),
                                 bkg_estimator=MedianBackground(),
                                 filter_size=(3, 3), exclude_percentile=50)
            residual = np.asarray(model.background, dtype=float)
            if residual.shape != corrected.shape or not np.isfinite(residual[valid]).all():
                raise ValueError(f"Invalid interpolated local background for {intensity_path}")
            corrected = corrected - residual
            details.update(mesh_arcmin=float(mesh_arcmin),
                           galaxy_mask_radius_arcsec=float(galaxy_mask_radius_arcsec),
                           fitted_mesh_count=usable, sky_fraction=float(np.mean(~bad & valid)),
                           residual_median=float(np.median(residual[valid])),
                           residual_rms_median=float(np.median(model.background_rms[valid])))
    corrected = np.asarray(corrected, dtype=float).copy()
    corrected[~valid] = np.nan
    if residual is not None:
        residual = residual.copy()
        residual[~valid] = np.nan
    details["valid_pixels"] = int(valid.sum())
    return corrected, residual, details
