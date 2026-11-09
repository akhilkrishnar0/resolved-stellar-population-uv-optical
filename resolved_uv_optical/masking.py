
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u
from astropy.visualization import simple_norm
from astropy.table import Table

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, EllipticalAperture


@dataclass
class MaskConfig:
    base_dir: str | Path
    out_dir: str | Path
    galaxy: str
    bands: Sequence[str] = field(default_factory=lambda: ["FUV", "NUV", "g", "r", "z"])
    file_pattern: str = "{galaxy}_{band}_psfmatch.fits"
    reference_band: str = "g"

    center_ra: Optional[float] = None
    center_dec: Optional[float] = None

    protect_galaxy: bool = True
    galaxy_pa_deg: float = 0.0
    galaxy_sma_arcsec: float = 180.0
    galaxy_smb_arcsec: float = 80.0

    dao_fwhm_pix: float = 3.0
    dao_threshold_sigma: float = 5.0

    use_gaia: bool = True
    gaia_search_radius_arcmin: float = 12.0
    match_radius_arcsec: float = 1.5
    pm_snr_limit: float = 3.0
    plx_snr_limit: float = 3.0
    allow_dao_fallback: bool = False

    auto_mask_radius_arcsec: float = 5.0
    mask_value: float = np.nan

    # Background subtraction. By default this follows the earlier UV stage:
    # subtract a scalar outer-annulus background only for FUV and NUV.
    do_bgsub: bool = True
    bgsub_bands: Sequence[str] = field(default_factory=lambda: ["FUV", "NUV"])
    bg_inner_frac: float = 0.70
    bg_sigma: float = 3.0
    bg_maxiters: int = 5
    bg_statistic: str = "median"  # "median" or "mean"
    write_bgsub_outputs: bool = True
    write_mask_fits: bool = True

    manual_circles_refpix: Sequence[Tuple[float, float, float]] = field(default_factory=list)
    manual_ellipses_refpix: Sequence[Tuple[float, float, float, float, float]] = field(default_factory=list)

    make_plots: bool = True
    png_dpi: int = 150

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)


def get_path(config: MaskConfig, band: str) -> Path:
    return config.base_dir / config.file_pattern.format(galaxy=config.galaxy, band=band)


def load_fits(path: str | Path) -> Tuple[np.ndarray, fits.Header, WCS]:
    data, header = fits.getdata(path, header=True)
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D FITS image: {path}")
    wcs = WCS(header).celestial
    return data, header, wcs


def image_center_sky(data: np.ndarray, wcs: WCS) -> SkyCoord:
    y0, x0 = np.array(data.shape) / 2.0
    return wcs.pixel_to_world(x0, y0)


def pixel_scale_arcsec(wcs: WCS) -> float:
    scale = np.abs(wcs.proj_plane_pixel_scales()[0])
    if hasattr(scale, "to"):
        return float(scale.to(u.arcsec).value)
    return float(scale * 3600.0)


def inside_protected_ellipse(
    coords: SkyCoord,
    center: SkyCoord,
    pa_deg: float,
    sma_arcsec: float,
    smb_arcsec: float,
) -> np.ndarray:
    east, north = center.spherical_offsets_to(coords)
    dra, ddec = east.to_value(u.arcsec), north.to_value(u.arcsec)
    pa = np.deg2rad(pa_deg)
    x = dra * np.cos(pa) + ddec * np.sin(pa)
    y = -dra * np.sin(pa) + ddec * np.cos(pa)
    rell = np.sqrt((x / sma_arcsec) ** 2 + (y / smb_arcsec) ** 2)
    return rell <= 1.0


def detect_sources_in_reference(data: np.ndarray, config: MaskConfig) -> Optional[Table]:
    finite = np.isfinite(data)
    if finite.sum() < 20:
        return None
    fill = np.nanmedian(data[finite])
    clean = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill)
    _, median, std = sigma_clipped_stats(clean, sigma=5.0, maxiters=5)
    if not np.isfinite(std) or std <= 0:
        return None
    daofind = DAOStarFinder(fwhm=config.dao_fwhm_pix, threshold=config.dao_threshold_sigma * std)
    sources = daofind(clean - median)
    if sources is None or len(sources) == 0:
        return None
    good = (
        np.isfinite(sources["xcentroid"]) &
        np.isfinite(sources["ycentroid"]) &
        (sources["xcentroid"] >= 0) &
        (sources["xcentroid"] < data.shape[1]) &
        (sources["ycentroid"] >= 0) &
        (sources["ycentroid"] < data.shape[0])
    )
    sources = sources[good]
    return sources if len(sources) > 0 else None


def query_gaia(center: SkyCoord, radius_arcmin: float) -> Table:
    from astroquery.gaia import Gaia
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    Gaia.ROW_LIMIT = -1
    job = Gaia.cone_search_async(coordinate=center, radius=radius_arcmin * u.arcmin)
    return job.get_results()


def select_gaia_stars(gaia: Table, config: MaskConfig) -> Table:
    n = len(gaia)
    if n == 0:
        return gaia
    pmra = np.asarray(gaia["pmra"], dtype=float)
    pmdec = np.asarray(gaia["pmdec"], dtype=float)
    pmra_err = np.asarray(gaia["pmra_error"], dtype=float)
    pmdec_err = np.asarray(gaia["pmdec_error"], dtype=float)
    pm = np.sqrt(pmra ** 2 + pmdec ** 2)
    pm_err = np.sqrt(pmra_err ** 2 + pmdec_err ** 2)
    pm_snr = np.zeros(n, dtype=float)
    ok_pm = np.isfinite(pm) & np.isfinite(pm_err) & (pm_err > 0)
    pm_snr[ok_pm] = pm[ok_pm] / pm_err[ok_pm]
    parallax = np.asarray(gaia["parallax"], dtype=float)
    parallax_err = np.asarray(gaia["parallax_error"], dtype=float)
    plx_snr = np.zeros(n, dtype=float)
    ok_plx = np.isfinite(parallax) & np.isfinite(parallax_err) & (parallax_err > 0)
    plx_snr[ok_plx] = np.abs(parallax[ok_plx]) / parallax_err[ok_plx]
    return gaia[(pm_snr > config.pm_snr_limit) | (plx_snr > config.plx_snr_limit)]


def make_circular_mask(shape: Tuple[int, int], positions: np.ndarray, radius_pix: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if positions is None or len(positions) == 0:
        return mask
    apertures = CircularAperture(positions, r=radius_pix)
    for ap in apertures:
        img = ap.to_mask(method="center").to_image(shape)
        if img is not None:
            mask |= img.astype(bool)
    return mask

def estimate_outer_annulus_background(data, inner_frac=0.70, sigma=3.0, maxiters=5, statistic="median"):
    data = np.asarray(data, dtype=float)

    ny, nx = data.shape
    yy, xx = np.indices(data.shape)

    xc = 0.5 * (nx - 1)
    yc = 0.5 * (ny - 1)

    rr = np.sqrt((xx - xc)**2 + (yy - yc)**2)
    rmax = np.nanmax(rr)

    annulus = rr >= inner_frac * rmax
    vals = data[annulus]
    vals = vals[np.isfinite(vals)]

    if len(vals) < 100:
        return np.nan, np.nan, np.nan, len(vals)

    clipped = sigma_clip(vals, sigma=sigma, maxiters=maxiters)
    good = np.asarray(clipped.data)[~np.ma.getmaskarray(clipped)]

    if len(good) < 50:
        return np.nan, np.nan, np.nan, len(good)

    bg_mean = float(np.nanmean(good))
    bg_median = float(np.nanmedian(good))
    bg_std = float(np.nanstd(good))

    if statistic not in ("mean", "median"):
        raise ValueError("bg_statistic must be mean or median")
    if statistic == "mean":
        bg = bg_mean
    else:
        bg = bg_median

    return bg, bg_mean, bg_std, len(good)

def make_manual_mask_on_reference(shape: Tuple[int, int], config: MaskConfig) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for x, y, r in config.manual_circles_refpix:
        img = CircularAperture((x, y), r=r).to_mask(method="center").to_image(shape)
        if img is not None:
            mask |= img.astype(bool)
    for x, y, a, b, theta_deg in config.manual_ellipses_refpix:
        img = EllipticalAperture((x, y), a=a, b=b, theta=np.deg2rad(theta_deg)).to_mask(method="center").to_image(shape)
        if img is not None:
            mask |= img.astype(bool)
    return mask


def reproject_mask_by_wcs(ref_mask: np.ndarray, ref_wcs: WCS, target_wcs: WCS, target_shape: Tuple[int, int]) -> np.ndarray:
    yy, xx = np.indices(target_shape)
    sky = target_wcs.pixel_to_world(xx, yy)
    x_ref, y_ref = ref_wcs.world_to_pixel(sky)
    finite = np.isfinite(x_ref) & np.isfinite(y_ref)
    x_ref = np.rint(np.where(finite, x_ref, -1)).astype(int)
    y_ref = np.rint(np.where(finite, y_ref, -1)).astype(int)
    valid = (
        finite & (x_ref >= 0) & (x_ref < ref_mask.shape[1]) &
        (y_ref >= 0) & (y_ref < ref_mask.shape[0])
    )
    target_mask = np.zeros(target_shape, dtype=bool)
    target_mask[valid] = ref_mask[y_ref[valid], x_ref[valid]]
    return target_mask


def save_diagnostic(data: np.ndarray, wcs: WCS, mask: np.ndarray, outpng: str | Path, title: str, dpi: int = 200) -> None:
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(projection=wcs)
    try:
        norm = simple_norm(data, "log", percent=99.5)
    except Exception:
        norm = simple_norm(data, "linear", percent=99.5)
    ax.imshow(data, origin="lower", cmap="gray", norm=norm)
    if np.any(mask):
        ax.contour(mask.astype(int), levels=[0.5], colors="red", linewidths=0.7)
    ax.set_title(title)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    plt.tight_layout()
    plt.savefig(outpng, dpi=dpi)
    plt.close(fig)


def build_reference_mask(config: MaskConfig) -> Dict[str, Any]:
    ref_path = get_path(config, config.reference_band)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    ref_data, ref_header, ref_wcs = load_fits(ref_path)
    center = image_center_sky(ref_data, ref_wcs) if config.center_ra is None or config.center_dec is None else SkyCoord(config.center_ra * u.deg, config.center_dec * u.deg, frame="icrs")
    sources = detect_sources_in_reference(ref_data, config)
    dao_all_count = 0
    dao_outside_count = 0
    gaia_error = None
    gaia_count = 0
    gaia_star_count = 0
    matched_count = 0
    fallback_used = False
    matched_star_coords = SkyCoord([], [], unit="deg", frame="icrs")
    source_coords_outside = SkyCoord([], [], unit="deg", frame="icrs")
    if sources is not None:
        dao_all_count = len(sources)
        source_coords = ref_wcs.pixel_to_world(np.asarray(sources["xcentroid"]), np.asarray(sources["ycentroid"]))
        if config.protect_galaxy:
            keep = ~inside_protected_ellipse(source_coords, center, config.galaxy_pa_deg, config.galaxy_sma_arcsec, config.galaxy_smb_arcsec)
        else:
            keep = np.ones(len(source_coords), dtype=bool)
        source_coords_outside = source_coords[keep]
        dao_outside_count = len(source_coords_outside)
        if config.use_gaia and dao_outside_count > 0:
            try:
                gaia = query_gaia(center, config.gaia_search_radius_arcmin)
                gaia_count = len(gaia)
                gaia_stars = select_gaia_stars(gaia, config)
                gaia_star_count = len(gaia_stars)
                if gaia_star_count > 0:
                    gaia_coords = SkyCoord(gaia_stars["ra"], gaia_stars["dec"], unit="deg", frame="icrs")
                    _, d2d, _ = match_coordinates_sky(source_coords_outside, gaia_coords)
                    good = d2d < config.match_radius_arcsec * u.arcsec
                    matched_star_coords = source_coords_outside[good]
                    matched_count = len(matched_star_coords)
            except Exception as exc:
                gaia_error = str(exc)
                if not config.allow_dao_fallback:
                    raise RuntimeError("Gaia query/matching failed; masking stopped") from exc
                print(f"[WARN] Gaia query/matching failed: {exc}")
        if len(matched_star_coords) == 0 and config.allow_dao_fallback:
            matched_star_coords = source_coords_outside
            matched_count = len(matched_star_coords)
            fallback_used = True
    x_ref, y_ref = ref_wcs.world_to_pixel(matched_star_coords)
    good_pix = (
        np.isfinite(x_ref) & np.isfinite(y_ref) &
        (x_ref >= 0) & (x_ref < ref_data.shape[1]) &
        (y_ref >= 0) & (y_ref < ref_data.shape[0])
    )
    positions_ref = np.transpose([x_ref[good_pix], y_ref[good_pix]]) if np.any(good_pix) else np.empty((0, 2))
    radius_pix = config.auto_mask_radius_arcsec / pixel_scale_arcsec(ref_wcs)
    auto_mask = make_circular_mask(ref_data.shape, positions_ref, radius_pix)
    manual_mask = make_manual_mask_on_reference(ref_data.shape, config)
    if config.protect_galaxy:
        yy, xx = np.indices(ref_data.shape)
        sky_all = ref_wcs.pixel_to_world(xx, yy)
        protect_mask = inside_protected_ellipse(sky_all, center, config.galaxy_pa_deg, config.galaxy_sma_arcsec, config.galaxy_smb_arcsec)
        ref_mask = (auto_mask & ~protect_mask) | manual_mask
    else:
        ref_mask = auto_mask | manual_mask
    table = Table()
    table["x_ref"] = positions_ref[:, 0] if len(positions_ref) else []
    table["y_ref"] = positions_ref[:, 1] if len(positions_ref) else []
    if len(positions_ref):
        sky_used = ref_wcs.pixel_to_world(positions_ref[:, 0], positions_ref[:, 1])
        table["ra_deg"] = sky_used.ra.deg
        table["dec_deg"] = sky_used.dec.deg
    else:
        table["ra_deg"] = []
        table["dec_deg"] = []
    return {
        "ref_path": ref_path,
        "ref_data": ref_data,
        "ref_header": ref_header,
        "ref_wcs": ref_wcs,
        "center": center,
        "ref_mask": ref_mask,
        "auto_mask": auto_mask,
        "manual_mask": manual_mask,
        "source_table": table,
        "stats": {
            "gaia_error": gaia_error,
            "dao_all": dao_all_count,
            "dao_outside_protected_region": dao_outside_count,
            "gaia_all": gaia_count,
            "gaia_pm_or_parallax_stars": gaia_star_count,
            "final_star_positions": matched_count,
            "dao_fallback_used": fallback_used,
            "mask_radius_pix_reference": radius_pix,
        },
    }


def run_masking(config: MaskConfig) -> Dict[str, Any]:
    missing = [str(get_path(config, b)) for b in config.bands if not get_path(config, b).is_file()]
    if missing:
        raise FileNotFoundError("Missing masking inputs: " + ", ".join(missing))
    result = build_reference_mask(config)
    ref_data = result["ref_data"]
    ref_header = result["ref_header"]
    ref_wcs = result["ref_wcs"]
    ref_mask = result["ref_mask"]

    ref_mask_fits = config.out_dir / f"{config.galaxy}_{config.reference_band}_reference_star_mask.fits"
    source_table_csv = config.out_dir / f"{config.galaxy}_{config.reference_band}_reference_star_sources.csv"

    fits.writeto(ref_mask_fits, ref_mask.astype(np.uint8), ref_header, overwrite=True)
    result["source_table"].write(source_table_csv, overwrite=True)

    if config.make_plots:
        save_diagnostic(
            ref_data, ref_wcs, ref_mask,
            config.out_dir / f"{config.galaxy}_{config.reference_band}_reference_mask_check.png",
            f"{config.galaxy} {config.reference_band}-band reference mask",
            dpi=config.png_dpi,
        )

    saved = []
    bgsub_bands = {str(b).upper() for b in config.bgsub_bands}

    for band in config.bands:
        path = get_path(config, band)
        if not path.exists():
            print(f"[MISS] {path}")
            continue

        data, header, wcs = load_fits(path)
        band_mask = reproject_mask_by_wcs(ref_mask, ref_wcs, wcs, data.shape)

        # Original point-source-masked output: same as before.
        masked = data.copy().astype(float)
        masked[band_mask] = config.mask_value

        header_masked = header.copy()
        header_masked["MASKREF"] = (config.reference_band, "Mask reference band")
        header_masked["MASKRAD"] = (float(config.auto_mask_radius_arcsec), "Auto mask radius arcsec")
        header_masked["PROTSMA"] = (float(config.galaxy_sma_arcsec), "Protected SMA arcsec")
        header_masked["PROTSMB"] = (float(config.galaxy_smb_arcsec), "Protected SMB arcsec")
        header_masked["PROTPA"] = (float(config.galaxy_pa_deg), "Protected PA deg")

        out_fits = config.out_dir / f"{config.galaxy}_{band}_pointmasked.fits"
        out_mask = config.out_dir / f"{config.galaxy}_{band}_pointmask.fits"
        fits.writeto(out_fits, masked, header_masked, overwrite=True)

        if config.write_mask_fits:
            fits.writeto(out_mask, band_mask.astype(np.uint8), header_masked, overwrite=True)

        bgsub_fits = None
        bg_value = np.nan
        bg_mean = np.nan
        bg_std = np.nan
        bg_npix = 0
        bg_status = "not_requested"

        # Additional background-subtracted output.
        # For FUV/NUV by default, estimate a scalar residual background
        # from the outer annulus after excluding masked source pixels.
        if config.do_bgsub and config.write_bgsub_outputs:
            bgsub_fits = config.out_dir / f"{config.galaxy}_{band}_pointmasked_bgsub.fits"
            header_bg = header_masked.copy()

            if str(band).upper() in bgsub_bands and bool(header.get("BGSUB", False)):
                bgsub = masked.copy()
                header_bg["BGSUB"] = (True, "Background already corrected upstream")
                header_bg["BGSRC"] = (str(header.get("BGMODE", "upstream")), "Upstream correction")
                bg_status = "already_corrected_no_repeat"
            elif str(band).upper() in bgsub_bands:
                bg_data = data.copy().astype(float)
                bg_data[band_mask] = np.nan
                bg_value, bg_mean, bg_std, bg_npix = estimate_outer_annulus_background(
                    bg_data,
                    inner_frac=config.bg_inner_frac,
                    sigma=config.bg_sigma,
                    maxiters=config.bg_maxiters,
                    statistic=config.bg_statistic,
                )

                if np.isfinite(bg_value):
                    bgsub = masked - bg_value
                    header_bg["BGSUB"] = (True, "Background subtracted after point-source masking")
                    header_bg["BGSUBMED"] = (float(bg_value), "Subtracted outer-annulus background")
                    header_bg["BGMEAN"] = (float(bg_mean), "Sigma-clipped mean in outer annulus")
                    header_bg["BGSTD"] = (float(bg_std), "Sigma-clipped std in outer annulus")
                    header_bg["BGNPIX"] = (int(bg_npix), "Pixels used for background estimate")
                    header_bg["BGINFR"] = (float(config.bg_inner_frac), "Outer annulus inner radius fraction")
                    header_bg["BGSIGMA"] = (float(config.bg_sigma), "Sigma clipping threshold")
                    header_bg["BGSRC"] = ("outer_annulus", "Background-estimation method")
                    header_bg["BGSTAT"] = (str(config.bg_statistic), "Background statistic")
                    bg_status = "bgsub_written"
                else:
                    bgsub = masked.copy()
                    header_bg["BGSUB"] = (False, "Background estimate failed; masked image copied")
                    header_bg["BGSUBMED"] = (None, "No finite background estimate")
                    header_bg["BGNPIX"] = (int(bg_npix), "Pixels retained after clipping")
                    header_bg["BGSRC"] = ("outer_annulus_failed", "Background-estimation method")
                    bg_status = "background_failed_masked_copied"
            else:
                # Keep a complete bgsub file set. Optical bands are copied unchanged,
                # matching the earlier UV-only background-subtraction workflow.
                bgsub = masked.copy()
                header_bg["BGSUB"] = (False, "Band copied unchanged; bgsub only applied to selected bands")
                header_bg["BGSRC"] = ("not_applied_to_band", "Background-estimation method")
                bg_status = "copied_no_bgsub"

            fits.writeto(bgsub_fits, bgsub, header_bg, overwrite=True)

        if config.make_plots:
            out_png = config.out_dir / f"{config.galaxy}_{band}_mask_check.png"
            save_diagnostic(data, wcs, band_mask, out_png, f"{config.galaxy} {band}-band applied mask", dpi=config.png_dpi)

        saved.append({
            "band": band,
            "input": str(path),
            "masked_fits": str(out_fits),
            "bgsub_fits": str(bgsub_fits) if bgsub_fits is not None else None,
            "mask_fits": str(out_mask) if config.write_mask_fits else None,
            "background": float(bg_value) if np.isfinite(bg_value) else np.nan,
            "background_mean": float(bg_mean) if np.isfinite(bg_mean) else np.nan,
            "background_std": float(bg_std) if np.isfinite(bg_std) else np.nan,
            "background_npix": int(bg_npix),
            "background_status": bg_status,
        })

        print(f"[SAVE] {out_fits}")
        if bgsub_fits is not None:
            print(f"[SAVE] {bgsub_fits}  status={bg_status}  bg={bg_value:.6g}")

    result["saved"] = saved
    result["reference_mask_fits"] = str(ref_mask_fits)
    result["source_table_csv"] = str(source_table_csv)

    print("\nSummary")
    print("-------")
    print(f"Galaxy: {config.galaxy}")
    print(f"Reference: {result['ref_path']}")
    print(f"Centre: RA={result['center'].ra.deg:.6f}, Dec={result['center'].dec.deg:.6f}")
    for key, val in result["stats"].items():
        print(f"{key}: {val}")
    print(f"Reference mask: {ref_mask_fits}")
    print(f"Source table: {source_table_csv}")

    return result

