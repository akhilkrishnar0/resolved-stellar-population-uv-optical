"""Circular FUV SFR profiles ported from the supplied Sci3 notebook.

Numerical functions retain the notebook methods and constants. No extinction or
inclination corrections or propagated uncertainties are introduced here.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.visualization import ImageNormalize, ZScaleInterval, AsinhStretch
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.aperture import CircularAperture
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
BANDS = ["FUV", "NUV", "g", "r", "z"]
BAND_ZP = {"FUV":18.82, "NUV":20.08, "g":22.5, "r":22.5, "z":22.5}
SIGMA_SFR_THRESHOLD = 1e-2
FUV_SFR_COEFF = 1.4e-28
MANUAL_FUV_BACKGROUND = {"UGC12845":0.00065, "UGC1455":0.00085}
R25_OVERRIDES_KPC = {"UGC9024":2.2}
STYLE = {"font.size":11, "axes.labelsize":12, "axes.linewidth":1.1,
    "xtick.direction":"in", "ytick.direction":"in", "xtick.top":True,
    "ytick.right":True, "pdf.fonttype":42, "ps.fonttype":42}


def get_data_wcs(path):
    """Read a FITS image and return data, header, and celestial WCS."""
    with fits.open(path) as hdul:
        hdu = hdul[0]
        data = np.squeeze(hdu.data).astype(float)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D science image: {path}")
        header = hdu.header.copy()
        wcs = WCS(header).celestial
    return data, header, wcs

def pixscale_arcsec(wcs):
    """Mean projected pixel scale in arcsec pixel^-1."""
    return float(np.mean(np.abs(proj_plane_pixel_scales(wcs)))) * 3600.0

def estimate_background(data):
    """Sigma-clipped global background estimate used by the radial-profile function."""
    finite = np.isfinite(data)
    if finite.sum() < 100:
        return 0.0, 0.0
    _, median, std = sigma_clipped_stats(data[finite], sigma=3.0, maxiters=5)
    return float(median), float(std)

def subtract_uv_background(data, center_xy=None, inner_frac=0.70, sigma=3.0):
    """
    Subtract the mean UV background estimated from the outer part of the cutout.

    The original notebook used the outer 30% of the image, defined by radii greater
    than inner_frac times the maximum distance from the adopted centre. The mean
    of that sigma-clipped background region is subtracted from the UV image.
    """
    data = np.asarray(data, dtype=float)
    ny, nx = data.shape
    yy, xx = np.indices(data.shape)

    if center_xy is None:
        x0, y0 = nx / 2, ny / 2
    else:
        x0, y0 = center_xy

    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    rmax = np.nanmax(rr)
    bg_mask = (rr > inner_frac * rmax) & np.isfinite(data)

    if bg_mask.sum() < 100:
        bg_mask = np.isfinite(data)

    mean_bg, median_bg, std_bg = sigma_clipped_stats(data[bg_mask], sigma=sigma, maxiters=5)
    data_sub = data - mean_bg
    return data_sub, float(mean_bg), float(median_bg), float(std_bg), int(bg_mask.sum())

def unit_mode(header, band):
    """Return the assumed flux unit and AB zeropoint for each band."""
    bunit = str(header.get("BUNIT", "")).lower()
    if "nanomag" in bunit or band in ["g", "r", "z"]:
        return "nanomaggy", 22.5
    if band in ["FUV", "NUV"]:
        return "cps", BAND_ZP[band]
    return "native", np.nan

def mu_from_mean(mean_flux_per_pix, pixarea_arcsec2, zp):
    """Convert mean flux per pixel to AB mag arcsec^-2."""
    flux_arcsec2 = mean_flux_per_pix / pixarea_arcsec2
    mu = np.full_like(flux_arcsec2, np.nan, dtype=float)
    ok = np.isfinite(flux_arcsec2) & (flux_arcsec2 > 0) & np.isfinite(zp)
    mu[ok] = zp - 2.5 * np.log10(flux_arcsec2[ok])
    return mu

def radial_profile(data, wcs, center_sky, galaxy, band, distance_mpc, size_arcsec, n_annuli=30):
    """Measure circular radial profiles in fixed-width annuli."""
    pixscale = pixscale_arcsec(wcs)
    pixarea = pixscale ** 2
    x0, y0 = wcs.world_to_pixel(center_sky)

    rmax_arcsec = size_arcsec / 2.0
    edges_arcsec = np.linspace(0, rmax_arcsec, n_annuli + 1)
    edges_pix = edges_arcsec / pixscale

    yy, xx = np.indices(data.shape)
    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

    # Preserve the original profile-level background treatment.
    bkg, bkg_std = estimate_background(data)
    img = data - bkg

    rows = []
    kpc_per_arcsec = distance_mpc * 1000.0 / 206265.0

    for i in range(n_annuli):
        r1, r2 = edges_pix[i], edges_pix[i + 1]
        mask = (rr >= r1) & (rr < r2) & np.isfinite(img)
        vals = img[mask]

        n_clipped = int(sigma_clip(vals, sigma=3.0, maxiters=5).count()) if vals.size else 0
        if vals.size < 10:
            mean, median, std, sumflux = np.nan, np.nan, np.nan, np.nan
        else:
            mean, median, std = sigma_clipped_stats(vals, sigma=3.0, maxiters=5)
            sumflux = float(np.nansum(vals))

        rmid_arcsec = 0.5 * (edges_arcsec[i] + edges_arcsec[i + 1])
        rows.append({
            "galaxy": galaxy,
            "band": band,
            "radius_inner_arcsec": edges_arcsec[i],
            "radius_outer_arcsec": edges_arcsec[i + 1],
            "radius_arcsec": rmid_arcsec,
            "radius_kpc": rmid_arcsec * kpc_per_arcsec,
            "npix": int(vals.size),
            "npix_clipped": n_clipped,
            "mean_flux_sem": float(std)/np.sqrt(n_clipped) if n_clipped > 1 else np.nan,
            "mean_flux_per_pix": float(mean),
            "median_flux_per_pix": float(median),
            "std_flux_per_pix": float(std),
            "sum_flux": sumflux,
            "background": bkg,
            "background_std": bkg_std,
            "pixscale_arcsec": pixscale,
        })

    df = pd.DataFrame(rows)
    mode, zp = unit_mode({}, band)
    df["mu_AB_arcsec2"] = mu_from_mean(df["mean_flux_per_pix"].values, pixarea, zp)
    df["zeropoint_used"] = zp
    df["flux_unit_assumed"] = mode
    return df, x0, y0, edges_pix

def smooth_profile(x, y, ngrid=400, sigma=2.0):
    """PCHIP-interpolate and Gaussian-smooth a radial profile."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]

    order = np.argsort(x)
    x, y = x[order], y[order]
    _, idx = np.unique(x, return_index=True)
    x, y = x[idx], y[idx]

    if len(x) < 4:
        return None, None

    xgrid = np.linspace(x.min(), x.max(), ngrid)
    ygrid = PchipInterpolator(x, y, extrapolate=False)(xgrid)

    if np.isfinite(ygrid).sum() > 10:
        ygrid = gaussian_filter1d(ygrid, sigma=sigma)

    return xgrid, ygrid

def get_kpc_per_arcsec(df):
    """Estimate the kpc/arcsec conversion from a profile table."""
    good = (df["radius_arcsec"] > 0) & (df["radius_kpc"] > 0)
    if good.sum() == 0:
        return np.nan
    return np.nanmedian(df.loc[good, "radius_kpc"] / df.loc[good, "radius_arcsec"])

def estimate_r25_from_g_r(gdf, rdf):
    """Estimate R25 from the transformed g,r-based B-band profile."""
    kpc_per_arcsec = get_kpc_per_arcsec(gdf)

    x_min = max(gdf["radius_kpc"].min(), rdf["radius_kpc"].min())
    x_max = min(gdf["radius_kpc"].max(), rdf["radius_kpc"].max())

    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return np.nan, np.nan, None, None

    xgrid = np.linspace(x_min, x_max, 600)
    g_interp = PchipInterpolator(gdf["radius_kpc"], gdf["mu_AB_arcsec2"], extrapolate=False)
    r_interp = PchipInterpolator(rdf["radius_kpc"], rdf["mu_AB_arcsec2"], extrapolate=False)

    mu_g = gaussian_filter1d(g_interp(xgrid), sigma=2)
    mu_r = gaussian_filter1d(r_interp(xgrid), sigma=2)
    mu_B = mu_g + 0.313 * (mu_g - mu_r) + 0.227

    good = np.isfinite(xgrid) & np.isfinite(mu_B)
    if good.sum() < 10:
        return np.nan, np.nan, xgrid, mu_B

    x = xgrid[good]
    y = mu_B[good]
    diff = y - 25.0
    cross = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]

    if len(cross) == 0:
        return np.nan, np.nan, xgrid, mu_B

    i = cross[0]
    r25_kpc = x[i] + (25.0 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    r25_arcsec = r25_kpc / kpc_per_arcsec if np.isfinite(kpc_per_arcsec) and kpc_per_arcsec > 0 else np.nan
    return r25_kpc, r25_arcsec, xgrid, mu_B

def mu_fuv_to_sigma_sfr(mu_fuv):
    """Convert FUV AB mag arcsec^-2 to SFR surface density in Msun Gyr^-1 pc^-2."""
    mu_fuv = np.asarray(mu_fuv, dtype=float)
    M_ab_pc2 = mu_fuv - 21.572
    Lnu_pc2 = 4.0 * np.pi * (10.0 * 3.085677581e18) ** 2 * 10 ** (-0.4 * (M_ab_pc2 + 48.6))
    sigma_sfr_yr_pc2 = FUV_SFR_COEFF * Lnu_pc2
    return sigma_sfr_yr_pc2 * 1e9

def smooth_log_profile(x, y, ngrid=500, sigma=2.0):
    """Interpolate a positive profile in log-space and return linear y-values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[good], y[good]

    order = np.argsort(x)
    x, y = x[order], y[order]
    _, idx = np.unique(x, return_index=True)
    x, y = x[idx], y[idx]

    if len(x) < 4:
        return None, None

    xgrid = np.linspace(x.min(), x.max(), ngrid)
    logy_grid = PchipInterpolator(x, np.log10(y), extrapolate=False)(xgrid)

    if np.isfinite(logy_grid).sum() > 10:
        logy_grid = gaussian_filter1d(logy_grid, sigma=sigma)

    return xgrid, 10 ** logy_grid

def find_uv_radius(x, sigma_sfr, threshold=SIGMA_SFR_THRESHOLD):
    """Find the last smoothed-profile crossing of the adopted Sigma_SFR threshold."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(sigma_sfr, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[good], y[good]

    order = np.argsort(x)
    x, y = x[order], y[order]

    if len(x) < 4:
        return np.nan

    xg, yg = smooth_log_profile(x, y)
    if xg is None:
        return np.nan

    diff = np.log10(yg) - np.log10(threshold)
    crossing = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]

    if len(crossing) == 0:
        if np.nanmedian(diff) > 0:
            return np.nanmax(xg)
        return np.nan

    i = crossing[-1]
    x1, x2 = xg[i], xg[i + 1]
    y1, y2 = diff[i], diff[i + 1]
    return x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1)

def save_overlay(data, wcs, x0, y0, edges_pix, galaxy, band, fig_dir):
    """Save a diagnostic image showing the circular apertures used for each profile."""
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(projection=wcs)
    norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
    ax.imshow(data, origin="lower", cmap="gray", norm=norm)

    for r in edges_pix[1::3]:
        CircularAperture((x0, y0), r=r).plot(ax=ax, color="cyan", lw=0.8, alpha=0.75)

    CircularAperture((x0, y0), r=edges_pix[-1]).plot(ax=ax, color="red", lw=1.4, alpha=0.9)
    ax.scatter([x0], [y0], s=35, c="yellow", marker="+")
    ax.set_title(f"{galaxy} {band}: circular apertures")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{galaxy}_{band}_apertures.png", dpi=220)
    plt.close(fig)

def plot_profiles(galaxy, df, fig_dir):
    """Plot multi-band AB surface-brightness profiles for one galaxy."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for band in BANDS:
        sub = df[(df.galaxy == galaxy) & (df.band == band)].copy()
        sub = sub[np.isfinite(sub["mu_AB_arcsec2"])]
        if len(sub) == 0:
            continue
        ax.plot(sub["radius_kpc"], sub["mu_AB_arcsec2"], marker="o", ms=3, lw=1.4, label=band)

    ax.invert_yaxis()
    ax.set_xlabel("Radius [kpc]")
    ax.set_ylabel(r"Surface brightness $\mu_{\rm AB}$ [mag arcsec$^{-2}$]")
    ax.set_title(f"{galaxy}: radial surface-brightness profile")
    ax.legend(frameon=False, ncol=3)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{galaxy}_radial_profiles.png", dpi=250)
    plt.close(fig)

def clean_profile(df):
    columns = ["radius_kpc", "radius_arcsec", "mu_AB_arcsec2"]
    if not set(columns).issubset(df.columns):
        raise ValueError(f"Profile requires {columns}")
    optional = [c for c in ("Sigma_SFR_error", "mean_flux_per_pix", "mean_flux_sem") if c in df.columns]
    result = df[columns+optional].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan).dropna(subset=columns)
    return result.sort_values("radius_kpc").drop_duplicates("radius_kpc")


def run_profiles(catalog, base_dir=".", *, input_dir=None,
                 file_pattern="{galaxy}_{band}_pointmasked.fits", r25_overrides=None,
                 external_fuv_profiles=None, diagnostics=True, show=False,
                 error_shade=True, shade_alpha=0.18):
    """Measure profiles and save the notebook-style Sigma_SFR versus R/R25 plot.

    Input directory contains one subdirectory per galaxy. Original notebook
    external FUV tables for UGC12845/UGC1455 must be supplied explicitly if selected.
    Catalogue distances and angular sizes take precedence over notebook sample lists.
    """
    if not np.isfinite(shade_alpha) or not 0 <= shade_alpha <= 1:
        raise ValueError("shade_alpha must be between 0 and 1")
    base_dir = Path(base_dir).resolve()
    input_dir = base_dir/"masked" if input_dir is None else Path(input_dir).resolve()
    out = base_dir/"radial_profile_outputs"
    fig_dir, tab_dir = out/"figures", out/"tables"
    table = catalog.to_pandas() if hasattr(catalog, "to_pandas") else pd.DataFrame(catalog).copy()
    required = ["galaxy", "ra_deg", "dec_deg", "distance", "size"]
    if table.empty or not set(required).issubset(table.columns):
        raise ValueError(f"Provide a nonempty catalogue with {required}")
    if table.galaxy.isna().any():
        raise ValueError("Missing galaxy names")
    table["galaxy"] = table.galaxy.astype(str).str.replace(" ", "", regex=False)
    if table.galaxy.duplicated().any() or any(g in ("", ".", "..") or "/" in g or "\\" in g for g in table.galaxy):
        raise ValueError("Invalid or duplicate galaxy names")
    vals = table[required[1:]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(vals.to_numpy()).all() or not vals.ra_deg.between(0,360,inclusive="left").all() or not vals.dec_deg.between(-90,90).all() or (vals.distance<=0).any() or (vals["size"]<=0).any():
        raise ValueError("Invalid coordinates, distances or sizes")
    table[required[1:]] = vals
    overrides = dict(R25_OVERRIDES_KPC)
    overrides.update(r25_overrides or {})
    if any(not np.isfinite(v) or v <= 0 for v in overrides.values()):
        raise ValueError("R25 overrides must be positive finite kpc values")
    external = {} if external_fuv_profiles is None else external_fuv_profiles
    for galaxy in set(table.galaxy) & {"UGC12845", "UGC1455"}:
        if galaxy not in external:
            raise ValueError(f"Notebook used an external FUV profile for {galaxy}; supply external_fuv_profiles for this galaxy")
    missing = [str(input_dir/g/file_pattern.format(galaxy=g, band=b)) for g in table.galaxy for b in BANDS if not (input_dir/g/file_pattern.format(galaxy=g, band=b)).is_file()]
    missing += [str(p) for g,p in external.items() if g in set(table.galaxy) and not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Missing profile inputs: " + ", ".join(missing))
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tab_dir/"input_catalog.csv", index=False)
    report = {"status":"running", "input_dir":str(input_dir), "file_pattern":file_pattern,
        "catalog":table.to_dict("records"), "n_annuli":30,
        "error_shade":error_shade, "shade_alpha":shade_alpha,
        "error_method":"clipped pixel std/sqrt(N_clipped), independent-pixel statistical SEM only",
        "error_limits":"No background/calibration/covariance/R25 uncertainty; smoothed display envelope, not a fitted confidence interval", "zeropoints":BAND_ZP,
        "FUV_SFR_coefficient":FUV_SFR_COEFF, "threshold":SIGMA_SFR_THRESHOLD,
        "manual_fuv_background":MANUAL_FUV_BACKGROUND, "r25_overrides_kpc":overrides,
        "external_fuv_profiles":{k:str(v) for k,v in external.items()}, "skipped":{}}
    report_path = out/"profile_report.json"
    def save_report():
        report_path.write_text(json.dumps(report, indent=2, default=str)+"\n")
    save_report()
    profiles, r25_rows, uv_rows, curve_rows, curves = [], [], [], [], []
    try:
        with plt.rc_context(STYLE):
            for row in table.itertuples(index=False):
                galaxy = row.galaxy
                center = SkyCoord(row.ra_deg, row.dec_deg, unit="deg")
                bands = {}
                for band in BANDS:
                    path = input_dir/galaxy/file_pattern.format(galaxy=galaxy, band=band)
                    data, hdr, wcs = get_data_wcs(path)
                    if not np.isfinite(data).any():
                        raise ValueError(f"No finite data: {path}")
                    if hdr.get("BGSUB", False) not in (False, 0, "", None):
                        raise ValueError(f"Use pre-background-subtraction masked inputs to preserve notebook treatment: {path}")
                    x0, y0 = wcs.world_to_pixel(center)
                    if not np.isfinite([x0,y0]).all() or not 0 <= x0 < data.shape[1] or not 0 <= y0 < data.shape[0]:
                        raise ValueError(f"Catalogue centre outside {path}")
                    bg_mean = bg_median = bg_std = np.nan
                    bg_npix = 0
                    if band in ("FUV", "NUV"):
                        if band == "FUV" and galaxy in MANUAL_FUV_BACKGROUND:
                            bg_mean = bg_median = MANUAL_FUV_BACKGROUND[galaxy]
                            bg_std = 0.0
                            data = data-bg_mean
                        else:
                            data,bg_mean,bg_median,bg_std,bg_npix = subtract_uv_background(data, (x0,y0), inner_frac=.70, sigma=3.0)
                    prof,x0,y0,edges = radial_profile(data,wcs,center,galaxy,band,row.distance,row.size)
                    mode,zp = unit_mode(hdr,band)
                    prof["mu_AB_arcsec2"] = mu_from_mean(prof.mean_flux_per_pix.to_numpy(),prof.pixscale_arcsec.to_numpy()**2,zp)
                    prof["zeropoint_used"], prof["flux_unit_assumed"] = zp,mode
                    prof["uv_background_mean_subtracted"] = bg_mean
                    prof["uv_background_median"] = bg_median
                    prof["uv_background_std"] = bg_std
                    prof["uv_background_npix"] = bg_npix
                    prof["input_path"] = str(path)
                    flux = prof.mean_flux_per_pix.to_numpy()
                    sem = prof.mean_flux_sem.to_numpy()
                    prof["mu_error_approx"] = np.divide(2.5/np.log(10)*sem,flux,
                        out=np.full(len(prof),np.nan),where=flux>0)
                    if band == "FUV":
                        factor = float(mu_fuv_to_sigma_sfr(zp))/prof.pixscale_arcsec.to_numpy()**2
                        prof["Sigma_SFR"] = factor*flux
                        prof["Sigma_SFR_error"] = factor*sem
                    prof.to_csv(tab_dir/f"{galaxy}_{band}_radial_profile.csv",index=False)
                    bands[band] = clean_profile(prof)
                    profiles.append(prof)
                    if diagnostics:
                        save_overlay(data,wcs,x0,y0,edges,galaxy,band,fig_dir)
                if diagnostics:
                    plot_profiles(galaxy,pd.concat(profiles[-5:],ignore_index=True),fig_dir)
                derived,arcsec = np.nan,np.nan
                if len(bands["g"])>=4 and len(bands["r"])>=4:
                    derived,arcsec,_,_ = estimate_r25_from_g_r(bands["g"],bands["r"])
                r25 = overrides.get(galaxy,derived)
                r25_rows.append(dict(galaxy=galaxy,R25_kpc_from_gr_Bprofile=derived,
                    R25_arcsec_from_gr_Bprofile=arcsec,R25_kpc_used=r25,
                    method="manual override" if galaxy in overrides else "transformed B first crossing at 25"))
                fuv = bands["FUV"]
                if galaxy in external:
                    fuv = clean_profile(pd.read_csv(external[galaxy]))
                    fuv.to_csv(tab_dir/f"{galaxy}_external_FUV_profile_used.csv",index=False)
                sigma_sfr = mu_fuv_to_sigma_sfr(fuv.mu_AB_arcsec2.to_numpy())
                ru = find_uv_radius(fuv.radius_kpc.to_numpy(),sigma_sfr)
                rua = find_uv_radius(fuv.radius_arcsec.to_numpy(),sigma_sfr)
                physical_x, physical_y = smooth_log_profile(fuv.radius_kpc,sigma_sfr)
                status = "unavailable"
                if physical_x is not None:
                    diff = np.log10(physical_y)-np.log10(SIGMA_SFR_THRESHOLD)
                    status = "crossing" if np.any(np.sign(diff[:-1])!=np.sign(diff[1:])) else ("lower_limit_at_last_radius" if np.nanmedian(diff)>0 else "below_threshold")
                uv_rows.append(dict(galaxy=galaxy,R_UV_kpc=ru,R_UV_arcsec=rua,R25_kpc_used=r25,
                    R_UV_over_R25=ru/r25 if np.isfinite(r25) and r25>0 else np.nan,
                    radius_status=status,threshold_Msun_Gyr_pc2=SIGMA_SFR_THRESHOLD,FUV_SFR_coefficient=FUV_SFR_COEFF))
                if not np.isfinite(r25) or r25<=0:
                    report["skipped"][galaxy] = "No valid R25; omitted from normalized plot"
                    warnings.warn(f"{galaxy}: no R25; not plotted in normalized units",RuntimeWarning)
                    continue
                x,y = smooth_log_profile(fuv.radius_kpc.to_numpy()/r25,sigma_sfr)
                if x is None:
                    report["skipped"][galaxy] = "Fewer than four usable FUV points"
                    continue
                error = np.full_like(y,np.nan)
                if error_shade:
                    if "Sigma_SFR_error" in fuv:
                        raw_error = fuv.Sigma_SFR_error.to_numpy()
                    elif {"mean_flux_per_pix", "mean_flux_sem"}.issubset(fuv.columns):
                        raw_error = np.divide(sigma_sfr*fuv.mean_flux_sem.to_numpy(), fuv.mean_flux_per_pix.to_numpy(),
                            out=np.full(len(fuv),np.nan),where=fuv.mean_flux_per_pix.to_numpy()>0)
                    else:
                        raw_error = np.full(len(fuv),np.nan)
                    raw_x = fuv.radius_kpc.to_numpy()/r25
                    valid = np.isfinite(raw_error) & (raw_error>=0) & np.isfinite(raw_x)
                    # Do not bridge missing uncertainty intervals or extrapolate errors.
                    for indices in np.split(np.flatnonzero(valid), np.where(np.diff(np.flatnonzero(valid))>1)[0]+1):
                        if len(indices)<2:
                            continue
                        inside = (x>=raw_x[indices[0]]) & (x<=raw_x[indices[-1]])
                        error[inside] = np.maximum(0,PchipInterpolator(raw_x[indices],raw_error[indices])(x[inside]))
                    if not np.isfinite(error).any():
                        warnings.warn(f"{galaxy}: no usable uncertainty columns; line plotted without shading",RuntimeWarning)
                        report.setdefault("unshaded",[]).append(galaxy)
                curves.append((galaxy,x,y,error))
                curve_rows.append(pd.DataFrame({"galaxy":galaxy,"R_over_R25":x,"Sigma_SFR_Msun_Gyr_pc2":y,
                    "Sigma_SFR_error":error,"lower_1sigma":y-error,"upper_1sigma":y+error}))
            pd.concat(profiles,ignore_index=True).to_csv(tab_dir/"all_galaxies_all_bands_radial_profiles.csv",index=False)
            pd.DataFrame(r25_rows).to_csv(tab_dir/"R25_from_g_r_profiles.csv",index=False)
            pd.DataFrame(uv_rows).to_csv(tab_dir/"UV_radius_from_FUV_sigma_SFR.csv",index=False)
            if not curves:
                raise ValueError("No normalized curves available; inspect profile tables and R25 values")
            pd.concat(curve_rows,ignore_index=True).to_csv(tab_dir/"FUV_sigma_SFR_plotted_curves.csv",index=False)
            fig,ax = plt.subplots(figsize=(7,5))
            try:
                for galaxy,x,y,error in curves:
                    line, = ax.plot(x,y,lw=1.5,label=galaxy)
                    if error_shade:
                        lower,upper = y-error,y+error
                        # Retain the true bounds in CSV; clip only for log-axis display.
                        ax.fill_between(x,np.maximum(lower,1e-3),np.maximum(upper,1e-3),
                            where=np.isfinite(error),color=line.get_color(),alpha=shade_alpha,
                            linewidth=0,zorder=line.get_zorder()-1)
                if error_shade:
                    ax.text(.98,.02,"Shading: statistical SEM only",transform=ax.transAxes,
                        ha="right",va="bottom",fontsize=8,color=".35")
                ax.axhline(SIGMA_SFR_THRESHOLD,color="k",ls="--",lw=2,zorder=0)
                ax.axvline(1,color="k",ls=":",lw=2,zorder=0)
                ax.text(.4,SIGMA_SFR_THRESHOLD*1.08,r"$\Sigma_{\rm SFR}=10^{-2}\;M_\odot\,{\rm Gyr^{-1}\,pc^{-2}}$",
                    transform=ax.get_yaxis_transform(),ha="right",va="bottom",fontsize=12,color=".25")
                ax.set(xscale="log",yscale="log",xlim=(.2,10),ylim=(1e-3,20),
                    xlabel=r"$R/R_{25}$",ylabel=r"$\Sigma_{\rm SFR,FUV}\;(M_\odot\,{\rm Gyr^{-1}\,pc^{-2}})$")
                ax.minorticks_on()
                ax.legend(fontsize=10,ncol=3,loc="best",markerfirst=False,frameon=True)
                png = fig_dir/"all_galaxies_FUV_sigma_SFR_profiles.png"
                pdf = png.with_suffix(".pdf")
                fig.savefig(png,dpi=300,bbox_inches="tight")
                fig.savefig(pdf,bbox_inches="tight")
                if show:
                    plt.show()
            finally:
                plt.close(fig)
        report.update(status="complete",figure_png=str(png),figure_pdf=str(pdf),tables=str(tab_dir))
    except Exception as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_report()
    return report

