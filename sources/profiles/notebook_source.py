# Archived source cells from final-Sci3-SBP_fit_cleaned_with_final_panels.ipynb
# Historical standalone code; not imported or executed by the package.
# Core numerical and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# FITS/WCS and astronomical utilities
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, ZScaleInterval, AsinhStretch
from astropy.wcs.utils import proj_plane_pixel_scales
from astroquery.simbad import Simbad
from photutils.aperture import CircularAperture

# Smoothing and interpolation used for manuscript-style profiles and radii
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

# Image/PDF tools used only for the final multi-panel PDF compilation
from PIL import Image, ImageOps, ImageDraw

BASE = Path("1psfmatched_data_galaxies/masked_outputs")
OUT = Path("radial_profile_outputs")
FIG = OUT / "figures"
TAB = OUT / "tables"

for path in [FIG, TAB]:
    path.mkdir(parents=True, exist_ok=True)

BANDS = ["FUV", "NUV", "g", "r", "z"]
BAND_ZP = {"FUV": 18.82, "NUV": 20.08, "g": 22.5, "r": 22.5, "z": 22.5}
SIMBAD_ALIASES = {"Malin1": "Malin 1", "Malin2": "Malin 2"}

GALAXIES = [
    "UGC9024", "NGC6902", "NGC5905", "NGC4017", "NGC5533", "UGC4422", "UGC12845",
    "UGC1455", "UGC1382", "UGC6614", "UGC3968", "UGC6754", "NGC7589", "MCG-01-59-027",
    "UGC4985", "UGC12740", "UGC1922", "UGC905", "LEDA68495", "UGC4219", "LEDA135657",
    "UGC0568", "Malin2", "UGC1752", "LEDA135754", "LEDA83544", "Malin1", "LEDA135648",
]

DISTANCES_MPC = [
    33, 39.9, 47.9, 49, 54.5, 61.9, 69.7, 72.4, 82.4, 90.8, 96.9, 100.3,
    125.6, 143.2, 145.4, 150.3, 155.6, 163.8, 177.1, 177.6, 181.4, 188.5,
    192.8, 254.9, 304.8, 337.9, 339.9, 532.3,
]

SIZE_ARCSEC = [
    1000, 1000, 1000, 1000, 900, 800, 700, 700, 500, 550, 500, 500, 400, 350,
    350, 350, 300, 300, 300, 300, 300, 250, 250, 200, 200, 200, 200, 200,
]

# Retained for metadata consistency with the original notebook.
SIZE_KPC_GIVEN = [
    160.0, 193.4, 232.2, 237.6, 237.8, 240.2, 236.9, 254.6, 239.8, 242.4,
    234.7, 242.7, 243.6, 242.8, 246.8, 254.6, 226.4, 238.2, 257.1, 257.9,
    264.0, 228.3, 233.7, 247.0, 295.6, 328.0, 329.9, 516.1,
]

SIGMA_SFR_THRESHOLD = 1e-2
FUV_SFR_COEFF = 1.4e-28

MU_LIMIT = {"FUV": 31.5, "NUV": 30.5, "g": 30.0, "r": 29.5, "z": 28.5}
UV_CUT_AFTER_RUV_KPC = 2.0

BAND_COLORS = {
    "FUV": "tab:purple",
    "NUV": "tab:blue",
    "g": "tab:green",
    "r": "tab:red",
    "z": "tab:brown",
}


#UV_PROFILE_GALAXIES = ['UGC9024', 'NGC6902', 'NGC5905', 'UGC1382', 'UGC6614', 'UGC6754', 'NGC7589', 'MCG-01-59-027', 'UGC1922', 'LEDA68495', 'UGC4219', 'Malin2', 'Malin1', 'LEDA135648']
UV_PROFILE_GALAXIES = [
    "UGC9024", "NGC6902", "NGC5905", "NGC4017", "NGC5533", "UGC4422", "UGC12845",
    "UGC1455", "UGC1382", "UGC6614", "UGC3968", "UGC6754", "NGC7589", "MCG-01-59-027",
    "UGC4985", "UGC12740", "UGC1922", "UGC905", "LEDA68495", "UGC4219", "LEDA135657",
    "UGC0568", "Malin2", "UGC1752", "LEDA135754", "LEDA83544", "Malin1", "LEDA135648",
]

plt.rcParams.update({
    "font.size": 12,
    "axes.linewidth": 1.25,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

Simbad.TIMEOUT = 60
custom_simbad = Simbad()
custom_simbad.add_votable_fields("ra(d)", "dec(d)")


def resolve_simbad(name):
    """Resolve a galaxy name through SIMBAD and return coordinates in degrees."""
    qname = SIMBAD_ALIASES.get(name, name)
    tab = custom_simbad.query_object(qname)
    if tab is None or len(tab) == 0:
        print(f"[WARN] SIMBAD failed: {name}")
        return np.nan, np.nan
    return float(tab["RA_d"][0]), float(tab["DEC_d"][0])


def load_or_resolve_positions():
    """Load cached galaxy positions or query SIMBAD once and save the result."""
    posfile = TAB / "simbad_positions.csv"
    if posfile.exists():
        return pd.read_csv(posfile)

    rows = []
    for galaxy, distance, size_arcsec, size_kpc in zip(GALAXIES, DISTANCES_MPC, SIZE_ARCSEC, SIZE_KPC_GIVEN):
        ra, dec = resolve_simbad(galaxy)
        rows.append({
            "galaxy": galaxy,
            "distance_Mpc": distance,
            "size_arcsec": size_arcsec,
            "size_kpc_given": size_kpc,
            "ra_deg": ra,
            "dec_deg": dec,
        })
        print(galaxy, ra, dec)

    df = pd.DataFrame(rows)
    df.to_csv(posfile, index=False)
    return df


def get_data_wcs(path):
    """Read a FITS image and return data, header, and celestial WCS."""
    with fits.open(path) as hdul:
        hdu = hdul[0]
        data = np.squeeze(hdu.data).astype(float)
        header = hdu.header
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


def save_overlay(data, wcs, x0, y0, edges_pix, galaxy, band):
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
    fig.savefig(FIG / f"{galaxy}_{band}_apertures.png", dpi=220)
    plt.close(fig)

def plot_profiles(galaxy, df):
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
    fig.savefig(FIG / f"{galaxy}_radial_profiles.png", dpi=250)
    plt.close(fig)


def plot_native_flux_profiles(galaxy, df):
    """Plot radial profiles in the native FITS units after background subtraction."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for band in BANDS:
        sub = df[(df.galaxy == galaxy) & (df.band == band)].copy()
        sub = sub[np.isfinite(sub["mean_flux_per_pix"])]
        if len(sub) == 0:
            continue
        ax.plot(sub["radius_kpc"], sub["mean_flux_per_pix"], marker="o", ms=3, lw=1.4, label=band)

    ax.set_xlabel("Radius [kpc]")
    ax.set_ylabel("Background-subtracted mean flux per pixel")
    ax.set_title(f"{galaxy}: radial brightness profile, native FITS units")
    ax.legend(frameon=False, ncol=3)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / f"{galaxy}_native_flux_profiles.png", dpi=250)
    plt.close(fig)



posdf = load_or_resolve_positions()
all_profiles = []
summary = []

for _, row in posdf.iterrows():
    galaxy = row["galaxy"]

    if not np.isfinite(row["ra_deg"]) or not np.isfinite(row["dec_deg"]):
        continue

    center = SkyCoord(row["ra_deg"], row["dec_deg"], unit="deg")
    galaxy_profiles = []
    print(f"\nProcessing {galaxy}")

    for band in BANDS:
        #path = BASE / galaxy / f"{band}_psfmatch.fits"
        path = BASE / f"{galaxy}_{band}_pointmasked_bgfilled.fits"
        if not path.exists():
            print(f"[MISS] {path}")
            continue

        try:
            data, hdr, wcs = get_data_wcs(path)
            x0, y0 = wcs.world_to_pixel(center)

            bg_mean = np.nan
            bg_median = np.nan
            bg_std = np.nan
            bg_npix = np.nan

            # if band in ["FUV", "NUV"]:
            #     if galaxy=="UGC12845" and band =="FUV":
            #         data = data-0.001
            #     elif galaxy=="UGC1455" and band =="FUV":
            #         data = data-0.0007
            #     else:
            #         data, bg_mean, bg_median, bg_std, bg_npix = subtract_uv_background(data, center_xy=(x0, y0),
            #                                                                             inner_frac=0.70, sigma=3.0)
            #     print(
            #         f"[BG] {galaxy} {band}: mean={bg_mean:.6g}, median={bg_median:.6g}, "
            #         f"std={bg_std:.6g}, npix={bg_npix}"
            #     )
            #     hdr["BGMEAN"] = bg_mean
            #     hdr["BGMED"] = bg_median
            #     hdr["BGSTD"] = bg_std
            #     hdr["BGNPIX"] = bg_npix
            #     hdr["BGSUB"] = "outer-image mean subtracted"
            manual_fuv_bg={"UGC12845":0.00065,"UGC1455":0.00085}
            
            if band in ["FUV","NUV"]:
                if band=="FUV" and galaxy in manual_fuv_bg:
                    bg_mean=manual_fuv_bg[galaxy]; bg_median=bg_mean; bg_std=0.0; bg_npix=0
                    data=data-bg_mean
                    bg_note="manual FUV background subtracted"
                    print(f"[BG-MANUAL] {galaxy} {band}: mean={bg_mean:.6g}")
                else:
                    data,bg_mean,bg_median,bg_std,bg_npix=subtract_uv_background(
                        data,center_xy=(x0,y0),inner_frac=0.70,sigma=3.0)
                    bg_note="outer-image mean subtracted"
                    print(f"[BG] {galaxy} {band}: mean={bg_mean:.6g}, median={bg_median:.6g}, std={bg_std:.6g}, npix={bg_npix}")
            
                hdr["BGMEAN"]=float(bg_mean)
                hdr["BGMED"]=float(bg_median)
                hdr["BGSTD"]=float(bg_std)
                hdr["BGNPIX"]=int(bg_npix)
                hdr["BGSUB"]=bg_note

            prof, x0, y0, edges_pix = radial_profile(
                data, wcs, center, galaxy, band, row["distance_Mpc"], row["size_arcsec"])

            prof["uv_background_mean_subtracted"] = bg_mean
            prof["uv_background_median"] = bg_median
            prof["uv_background_std"] = bg_std
            prof["uv_background_npix"] = bg_npix

            mode, zp = unit_mode(hdr, band)
            prof["zeropoint_used"] = zp
            prof["flux_unit_assumed"] = mode
            prof["mu_AB_arcsec2"] = mu_from_mean(
                prof["mean_flux_per_pix"].values,
                prof["pixscale_arcsec"].values ** 2,
                zp,
            )

            prof.to_csv(TAB / f"{galaxy}_{band}_radial_profile.csv", index=False)
            save_overlay(data, wcs, x0, y0, edges_pix, galaxy, band)
            galaxy_profiles.append(prof)

            finite_mu = prof["mu_AB_arcsec2"].replace([np.inf, -np.inf], np.nan).dropna()
            summary.append({
                "galaxy": galaxy,
                "band": band,
                "distance_Mpc": row["distance_Mpc"],
                "size_arcsec": row["size_arcsec"],
                "rmax_kpc": 0.5 * row["size_arcsec"] * row["distance_Mpc"] * 1000 / 206265,
                "pixscale_arcsec": prof["pixscale_arcsec"].median(),
                "central_mu_AB_arcsec2": finite_mu.iloc[0] if len(finite_mu) > 0 else np.nan,
                "outer_mu_AB_arcsec2": finite_mu.iloc[-1] if len(finite_mu) > 0 else np.nan,
                "n_annuli": len(prof),
                "flux_unit_assumed": mode,
                "zeropoint_used": zp,
                "uv_background_mean_subtracted": bg_mean,
                "uv_background_median": bg_median,
                "uv_background_std": bg_std,
                "uv_background_npix": bg_npix,
            })

            print(f"[OK] {galaxy} {band}")

        except Exception as err:
            print(f"[FAIL] {galaxy} {band}: {err}")

    if len(galaxy_profiles) > 0:
        gdf = pd.concat(galaxy_profiles, ignore_index=True)
        all_profiles.append(gdf)
        plot_profiles(galaxy, gdf)
        plot_native_flux_profiles(galaxy, gdf)

if len(all_profiles) > 0:
    final = pd.concat(all_profiles, ignore_index=True)
    final.to_csv(TAB / "all_galaxies_all_bands_radial_profiles.csv", index=False)
    pd.DataFrame(summary).to_csv(TAB / "profile_summary_table.csv", index=False)
    print("\nSaved:")
    print(TAB / "all_galaxies_all_bands_radial_profiles.csv")
    print(TAB / "profile_summary_table.csv")
    print(FIG)
else:
    print("No profiles created.")

def load_profile(galaxy, band):
    """Load a saved radial profile for one galaxy and band."""
    path = TAB / f"{galaxy}_{band}_radial_profile.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    needed = ["radius_kpc", "radius_arcsec", "mu_AB_arcsec2"]
    if not all(col in df.columns for col in needed):
        return None

    df = df[needed].dropna()
    df = df[
        np.isfinite(df["radius_kpc"]) &
        np.isfinite(df["radius_arcsec"]) &
        np.isfinite(df["mu_AB_arcsec2"])
    ].copy()
    df = df.sort_values("radius_kpc")
    return df if len(df) >= 4 else None


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


R25_FIG = FIG / "profile_panels_manuscript"
R25_FIG.mkdir(parents=True, exist_ok=True)
r25_rows = []

for galaxy in GALAXIES:
    profiles = {}
    kpc_per_arcsec = np.nan

    for band in BANDS:
        df = load_profile(galaxy, band)
        if df is not None:
            profiles[band] = df
            if not np.isfinite(kpc_per_arcsec):
                kpc_per_arcsec = get_kpc_per_arcsec(df)

    if len(profiles) == 0:
        print(f"[SKIP] No usable profiles for {galaxy}")
        continue

    r25_kpc, r25_arcsec, rb_grid, mu_B = np.nan, np.nan, None, None
    if "g" in profiles and "r" in profiles:
        r25_kpc, r25_arcsec, rb_grid, mu_B = estimate_r25_from_g_r(profiles["g"], profiles["r"])

    fig, ax = plt.subplots(figsize=(5, 5))

    for band in BANDS:
        if band not in profiles:
            continue
        xgrid, ygrid = smooth_profile(profiles[band]["radius_kpc"], profiles[band]["mu_AB_arcsec2"])
        if xgrid is None:
            continue
        ax.plot(xgrid, ygrid, lw=2.2, color=BAND_COLORS.get(band), label=band, solid_capstyle="round")

    if rb_grid is not None and mu_B is not None and np.isfinite(mu_B).sum() > 10:
        ax.plot(rb_grid, mu_B, lw=1.8, color="black", ls="--", alpha=0.85, label=r"$B(g,r)$")

    if np.isfinite(r25_kpc):
        ax.axvline(r25_kpc, color="black", ls=":", lw=1.8)
        y0, y1 = ax.get_ylim()
        ax.text(r25_kpc, y1 + 0.08 * (y0 - y1), rf"$R_{{25}}={r25_kpc:.1f}$ kpc",
                rotation=90, va="bottom", ha="right", fontsize=11)

    ax.axhline(25.0, color="0.35", ls=":", lw=1.2, alpha=0.8)
    ax.text(0.985, 25.0, r"$\mu_B=25$", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=10, color="0.25")

    ax.set_xlabel("Radius [kpc]")
    ax.set_ylabel(r"Surface brightness $\mu_{\rm AB}$ [mag arcsec$^{-2}$]")
    ax.set_title(galaxy, fontsize=15, pad=10)
    ax.invert_yaxis()
    ax.minorticks_on()
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(loc="best", ncol=2, fontsize=10)

    if np.isfinite(kpc_per_arcsec) and kpc_per_arcsec > 0:
        secax = ax.secondary_xaxis("top", functions=(lambda x: x / kpc_per_arcsec, lambda x: x * kpc_per_arcsec))
        secax.set_xlabel("Radius [arcsec]")
        secax.minorticks_on()

    fig.savefig(R25_FIG / f"{galaxy}_all_bands_profile_manuscript.png", dpi=300)
    fig.savefig(R25_FIG / f"{galaxy}_all_bands_profile_manuscript.pdf")
    plt.close(fig)

    r25_rows.append({
        "galaxy": galaxy,
        "R25_kpc_from_gr_Bprofile": r25_kpc,
        "R25_arcsec_from_gr_Bprofile": r25_arcsec,
        "kpc_per_arcsec": kpc_per_arcsec,
        "method": "mu_B = mu_g + 0.313*(mu_g-mu_r) + 0.227; R25 at mu_B=25 mag/arcsec2",
    })

    print(f"[OK] {galaxy}: R25 = {r25_kpc:.2f} kpc, {r25_arcsec:.1f} arcsec")

r25_table = pd.DataFrame(r25_rows)
r25_table.to_csv(TAB / "R25_from_g_r_profiles.csv", index=False)

print("\nSaved manuscript figures to:")
print(R25_FIG)
print("\nSaved R25 table:")
print(TAB / "R25_from_g_r_profiles.csv")

def mu_fuv_to_sigma_sfr(mu_fuv):
    """Convert FUV AB mag arcsec^-2 to SFR surface density in Msun Gyr^-1 pc^-2."""
    mu_fuv = np.asarray(mu_fuv, dtype=float)
    M_ab_pc2 = mu_fuv - 21.572
    Lnu_pc2 = 4.0 * np.pi * (10.0 * 3.085677581e18) ** 2 * 10 ** (-0.4 * (M_ab_pc2 + 48.6))
    sigma_sfr_yr_pc2 = FUV_SFR_COEFF * Lnu_pc2
    return sigma_sfr_yr_pc2 * 1e9


def load_fuv_profile(galaxy):
    if galaxy in ["UGC12845","UGC1455"]:
        path=TAB/f"original_{galaxy}_FUV_radial_profile.csv"
    else:
        path=TAB/f"{galaxy}_FUV_radial_profile.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    required = ["radius_kpc", "radius_arcsec", "mu_AB_arcsec2"]
    if not all(col in df.columns for col in required):
        return None

    df = df[required].dropna()
    df = df[np.isfinite(df["radius_kpc"]) & np.isfinite(df["mu_AB_arcsec2"])].sort_values("radius_kpc")
    if len(df) < 4:
        return None

    df["Sigma_SFR"] = mu_fuv_to_sigma_sfr(df["mu_AB_arcsec2"])
    df = df[np.isfinite(df["Sigma_SFR"]) & (df["Sigma_SFR"] > 0)]
    return df if len(df) >= 4 else None


def load_r25_table():
    """Load the available R25 values and preserve the original column-priority logic."""
    path = TAB / "R25_from_g_r_profiles.csv"
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    possible_cols = [
        "R25_kpc", "R25_g_kpc", "R25_r_kpc", "R25_kpc_from_gr_Bprofile",
        "R25_kpc_from_g_profile", "R25_kpc_from_r_profile",
    ]

    r25 = {}
    for _, row in df.iterrows():
        vals = []
        for col in possible_cols:
            if col in df.columns and np.isfinite(row.get(col, np.nan)):
                vals.append(row[col])
        if len(vals) > 0:
            r25[row["galaxy"]] = np.nanmedian(vals)
    return r25


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

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

UV_PROFILE_GALAXIES = ['UGC9024', 'NGC6902',  'UGC1382', 'UGC6614', 'UGC6754', 'NGC7589', 'MCG-01-59-027', 'UGC1922', 'LEDA68495', 'UGC4219', 'Malin2', 'Malin1', 'LEDA135648']
# UV_PROFILE_GALAXIES = [
#     "UGC9024", "NGC6902", "NGC5905", "NGC4017", "NGC5533", "UGC4422", "UGC12845",
#     "UGC1455", "UGC1382", "UGC6614", "UGC3968", "UGC6754", "NGC7589", "MCG-01-59-027",
#     "UGC4985", "UGC12740", "UGC1922", "UGC905", "LEDA68495", "UGC4219", "LEDA135657",
#     "UGC0568", "Malin2", "UGC1752", "LEDA135754", "LEDA83544", "Malin1", "LEDA135648"]


r25_dict = load_r25_table()
uv_rows = []
fig, ax = plt.subplots(figsize=(7, 5))
xlabel = r"$R/R_{25}$"

for galaxy in UV_PROFILE_GALAXIES:
    df = load_fuv_profile(galaxy)
    if df is None:
        print(f"[SKIP] {galaxy}: no usable FUV profile")
        continue

    r25 = r25_dict.get(galaxy, np.nan)
    if galaxy == "UGC9024":
        r25 = 2.2
        print(r25)
    if np.isfinite(r25) and r25 > 0:
        x = df["radius_kpc"].values / r25
        xlabel = r"$R/R_{25}$"
    else:
        x = df["radius_kpc"].values
        xlabel = "Radius [kpc]"

    y = df["Sigma_SFR"].values
    xg, yg = smooth_log_profile(x, y)

    if xg is None:
        print(f"[SKIP] {galaxy}: smoothing failed")
        continue

    ax.plot(xg, yg, lw=1.5, label=galaxy)

    r_uv_kpc = find_uv_radius(df["radius_kpc"].values, df["Sigma_SFR"].values)
    r_uv_arcsec = find_uv_radius(df['radius_arcsec'].values, df["Sigma_SFR"].values)
    r_uv_norm = r_uv_kpc / r25 if np.isfinite(r_uv_kpc) and np.isfinite(r25) and r25 > 0 else np.nan

    uv_rows.append({
        "galaxy": galaxy,
        "R_UV_kpc": r_uv_kpc,
        "R25_kpc_used": r25,
        'R_UV_arcsec':r_uv_arcsec,
        "R_UV_over_R25": r_uv_norm,
        "threshold_Msun_Gyr_pc2": SIGMA_SFR_THRESHOLD,
        "FUV_SFR_coefficient": FUV_SFR_COEFF,
    })

    print(f"[OK] {galaxy}: R_UV = {r_uv_kpc:.2f} kpc,R_UV = {r_uv_arcsec:.2f} arcsec ,R_UV/R25 = {r_uv_norm:.2f}")

ax.axhline(SIGMA_SFR_THRESHOLD, color="k", ls="--", lw=2, zorder=0)
ax.axvline(1, color="k", ls=":", lw=2, zorder=0)

ax.text(
    0.4,
    SIGMA_SFR_THRESHOLD * 1.08,
    r"$\Sigma_{\rm SFR}=10^{-2}\;M_\odot\,{\rm Gyr^{-1}\,pc^{-2}}$",
    transform=ax.get_yaxis_transform(),
    ha="right",
    va="bottom",
    fontsize=12,
    color="0.25",
)

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(xlabel)
ax.set_ylabel(r"$\Sigma_{\rm SFR,FUV}\;(M_\odot\,{\rm Gyr^{-1}\,pc^{-2}})$")
ax.minorticks_on()
ax.legend(fontsize=10, ncol=3, loc="best",markerfirst=False,frameon=True)
ax.set_ylim(1e-3, 20)
ax.set_xlim(0.2, 10)

fig.savefig("final-SCi-plots/all_galaxies_FUV_sigma_SFR_profiles.png", dpi=300)
fig.savefig("final-SCi-plots/all_galaxies_FUV_sigma_SFR_profiles.pdf")
plt.show()

# uv_table = pd.DataFrame(uv_rows)
# uv_table.to_csv(TAB / "UV_radius_from_FUV_sigma_SFR.csv", index=False)

# print("\nSaved figure:")
# print(FIG / "all_galaxies_FUV_sigma_SFR_profiles.png")
# print("\nSaved UV-radius table:")
# print(TAB / "UV_radius_from_FUV_sigma_SFR.csv")









def cut_after_mu_limit(df, band):
    """Keep profile points only until the first radius fainter than the adopted band limit."""
    limit = MU_LIMIT[band]
    df = df.sort_values("radius_kpc").copy()
    bad = np.where(df["mu_AB_arcsec2"].values > limit)[0]
    if len(bad) > 0:
        df = df.iloc[:bad[0]]
    return df


def load_radius_table(path, cols):
    """Load a radius table using the first available valid column from cols."""
    if not path.exists():
        print(f"[WARN] Missing {path}")
        return {}

    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    out = {}

    for _, row in df.iterrows():
        for col in cols:
            if col in df.columns:
                value = row.get(col, np.nan)
                if np.isfinite(value) and value > 0:
                    out[row["galaxy"]] = value
                    break
    return out


FINAL_PANEL_DIR = FIG / "profile_panels_R25_RUV"
FINAL_PANEL_DIR.mkdir(parents=True, exist_ok=True)

r25_dict = load_radius_table(
    TAB / "R25_from_g_r_profiles.csv",
    ["R25_kpc", "R25_g_kpc", "R25_r_kpc", "R25_kpc_from_gr_Bprofile"],
)
ruv_dict = load_radius_table(
    TAB / "UV_radius_from_FUV_sigma_SFR.csv",
    ["R_UV_kpc", "RUV_kpc", "R_UV", "Ruv_kpc"],
)

summary_rows = []

for galaxy in GALAXIES:
    profiles = {}
    kpc_per_arcsec = np.nan

    for band in BANDS:
        df = load_profile(galaxy, band)
        if df is not None:
            profiles[band] = df
            if not np.isfinite(kpc_per_arcsec):
                kpc_per_arcsec = get_kpc_per_arcsec(df)

    if len(profiles) == 0:
        print(f"[SKIP] {galaxy}: no usable profiles")
        continue

    r25 = r25_dict.get(galaxy, np.nan)
    ruv = ruv_dict.get(galaxy, np.nan)
    fig, ax = plt.subplots(figsize=(5.3, 5.0))

    for band in BANDS:
        if band not in profiles:
            continue

        df = profiles[band].copy()
        if band in ["FUV", "NUV"] and np.isfinite(ruv):
            df = df[df["radius_kpc"] <= ruv + UV_CUT_AFTER_RUV_KPC]

        df = cut_after_mu_limit(df, band)
        if len(df) < 4:
            continue

        xgrid, ygrid = smooth_profile(df["radius_kpc"], df["mu_AB_arcsec2"], ngrid=500, sigma=1.8)
        if xgrid is None:
            continue

        ax.plot(xgrid, ygrid, lw=2.1, color=BAND_COLORS[band], label=band, solid_capstyle="round")

    if np.isfinite(r25):
        ax.axvline(r25, color="black", ls=":", lw=1.8, zorder=0)
        ax.text(r25, 0.98, rf"$R_{{25}}={r25:.1f}$ kpc", transform=ax.get_xaxis_transform(),
                rotation=90, ha="right", va="top", fontsize=10, color="black")

    if np.isfinite(ruv):
        ax.axvline(ruv, color="tab:blue", ls=":", lw=1.8, zorder=0)
        ax.text(ruv, 0.98, rf"$R_{{UV}}={ruv:.1f}$ kpc", transform=ax.get_xaxis_transform(),
                rotation=90, ha="left", va="top", fontsize=10, color="tab:blue")

    ax.set_xlabel("Radius [kpc]")
    ax.set_ylabel(r"$\mu_{\rm AB}$ [mag arcsec$^{-2}$]")
    ax.set_title(galaxy, fontsize=15, pad=8)
    ax.invert_yaxis()
    ax.minorticks_on()
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(loc="best", fontsize=9, ncol=2)

    if np.isfinite(kpc_per_arcsec) and kpc_per_arcsec > 0:
        secax = ax.secondary_xaxis("top", functions=(lambda x: x / kpc_per_arcsec, lambda x: x * kpc_per_arcsec))
        secax.set_xlabel("Radius [arcsec]")
        secax.minorticks_on()

    fig.savefig(FINAL_PANEL_DIR / f"{galaxy}_profiles_R25_RUV.png", dpi=300)
    fig.savefig(FINAL_PANEL_DIR / f"{galaxy}_profiles_R25_RUV.pdf")
    plt.close(fig)

    summary_rows.append({
        "galaxy": galaxy,
        "R25_kpc": r25,
        "RUV_kpc": ruv,
        "RUV_over_R25": ruv / r25 if np.isfinite(ruv) and np.isfinite(r25) and r25 > 0 else np.nan,
        "kpc_per_arcsec": kpc_per_arcsec,
    })

    print(f"[OK] {galaxy}: R25={r25:.2f} kpc, RUV={ruv:.2f} kpc")

summary = pd.DataFrame(summary_rows)
summary.to_csv(TAB / "profile_plot_R25_RUV_summary.csv", index=False)

print("\nSaved figures to:")
print(FINAL_PANEL_DIR)

IN = FIG / "profile_panels_R25_RUV"
OUT_PDF = FIG / "all_profile_panels_R25_RUV.pdf"

imgs = sorted(IN.glob("*.png"))
if len(imgs) == 0:
    raise FileNotFoundError(f"No PNG files found in {IN}")

cols = 3
thumb_w = 1200
pad = 80
label_h = 90
bg = "white"
pages = []

for i in range(0, len(imgs), cols):
    group = imgs[i:i + cols]
    thumbs = []

    for f in group:
        im = Image.open(f).convert("RGB")
        im = ImageOps.contain(im, (thumb_w, thumb_w))
        canvas = Image.new("RGB", (thumb_w, im.height + label_h), bg)
        canvas.paste(im, ((thumb_w - im.width) // 2, label_h))
        draw = ImageDraw.Draw(canvas)
        name = f.stem.replace("_profiles_R25_RUV", "")
        draw.text((30, 25), name, fill="black")
        thumbs.append(canvas)

    max_h = max(t.height for t in thumbs)
    page_w = cols * thumb_w + (cols + 1) * pad
    page_h = max_h + 2 * pad
    page = Image.new("RGB", (page_w, page_h), bg)

    for j, thumb in enumerate(thumbs):
        x = pad + j * (thumb_w + pad)
        y = pad
        page.paste(thumb, (x, y))

    pages.append(page)

pages[0].save(OUT_PDF, save_all=True, append_images=pages[1:], resolution=300)
print(f"Saved PDF: {OUT_PDF}")
print(f"Number of images: {len(imgs)}")

FINAL_MANUSCRIPT_DIR = FIG / "selected_galaxy_image_profile_panels"
FINAL_MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_FINAL_GALAXIES = UV_PROFILE_GALAXIES


# ============================================================
# BASIC HELPERS
# ============================================================

def get_position_row(galaxy):
    positions = load_or_resolve_positions()
    match = positions[positions["galaxy"] == galaxy]
    if len(match) == 0:
        return None
    return match.iloc[0]


def load_image_for_display(galaxy, band):
    row = get_position_row(galaxy)
    if row is None:
        return None, None, None, None, None

    path = BASE / f"{galaxy}_{band}_pointmasked.fits"
    if not path.exists():
        return None, None, None, None, None

    data, header, wcs = get_data_wcs(path)
    center = SkyCoord(row["ra_deg"], row["dec_deg"], unit="deg")
    x0, y0 = wcs.world_to_pixel(center)

    if band in ["FUV", "NUV"]:
        data, *_ = subtract_uv_background(
            data,
            center_xy=(x0, y0),
            inner_frac=0.70,
            sigma=3.0
        )

    return data, header, wcs, center, row


def image_norm(data):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    return ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())


def radius_kpc_to_pix(radius_kpc, kpc_per_arcsec, pixscale):
    if not np.isfinite(radius_kpc) or not np.isfinite(kpc_per_arcsec) or kpc_per_arcsec <= 0:
        return np.nan
    return (radius_kpc / kpc_per_arcsec) / pixscale


def radius_kpc_to_arcsec(radius_kpc, kpc_per_arcsec):
    if not np.isfinite(radius_kpc) or not np.isfinite(kpc_per_arcsec) or kpc_per_arcsec <= 0:
        return np.nan
    return radius_kpc / kpc_per_arcsec


# ============================================================
# SAME SKY-FOV CROPPING
# ============================================================

def crop_image_same_sky_fov(data, wcs, center, crop_radius_arcsec):
    """
    Crop around the same sky centre using the same angular crop radius.
    This makes FUV and g cover the same sky field of view.
    """
    x0, y0 = wcs.world_to_pixel(center)
    pixscale = pixscale_arcsec(wcs)

    if not np.isfinite(crop_radius_arcsec) or crop_radius_arcsec <= 0:
        return data, 0, 0, x0, y0, pixscale

    crop_radius_pix = crop_radius_arcsec / pixscale

    ny, nx = data.shape

    x1 = int(max(0, x0 - crop_radius_pix))
    x2 = int(min(nx, x0 + crop_radius_pix))
    y1 = int(max(0, y0 - crop_radius_pix))
    y2 = int(min(ny, y0 + crop_radius_pix))

    cropped = data[y1:y2, x1:x2]

    return cropped, x1, y1, x0, y0, pixscale


def crop_radius_for_galaxy(galaxy, ruv_kpc, kpc_per_arcsec):
    if not np.isfinite(ruv_kpc) or not np.isfinite(kpc_per_arcsec) or kpc_per_arcsec <= 0:
        return np.nan

    ruv_arcsec = ruv_kpc / kpc_per_arcsec

    crop_factor = {
        "UGC9024": 1.25,
        "NGC6902": 1.10,
        "NGC5905": 1.10,
        "UGC1382": 1.20,
        "UGC6614": 1.20,
        "UGC6754": 1.20,
        "NGC7589": 1.35,
        "MCG-01-59-027": 1.30,
        "UGC1922": 1.30,
        "LEDA68495": 1.35,
        "UGC4219": 1.25,
        "Malin2": 1.30,
        "Malin1": 1.45,
        "LEDA135648": 1.55,
    }

    return crop_factor.get(galaxy, 1.25) * ruv_arcsec


def lock_image_limits(ax, image):
    """
    Critical fix:
    after drawing circles/annuli, force axis limits back to the image size.
    This prevents large annuli from shrinking the displayed image.
    """
    ny, nx = image.shape
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_aspect("equal", adjustable="box")


# ============================================================
# OVERLAYS
# ============================================================

def overlay_radius(ax, x0, y0, r_pix, label, color, lw=2.0):
    if np.isfinite(r_pix) and r_pix > 0:
        CircularAperture((x0, y0), r=r_pix).plot(
            ax=ax,
            color=color,
            lw=lw,
            alpha=0.95
        )

        ax.text(
            0.04,
            0.94 if label == r"$R_{25}$" else 0.87,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color=color,
            weight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.65,
                pad=2.0
            ),
            zorder=30
        )


def overlay_fuv_annuli(ax, x0, y0, fuv_profile, pixscale, max_radius_pix=None):
    """
    Overlay FUV annuli. If max_radius_pix is given, only annuli within the
    displayed crop are drawn.
    """
    if fuv_profile is None or "radius_outer_arcsec" not in fuv_profile.columns:
        return

    annulus_radii = np.asarray(fuv_profile["radius_outer_arcsec"], dtype=float) / pixscale
    annulus_radii = annulus_radii[np.isfinite(annulus_radii) & (annulus_radii > 0)]

    if max_radius_pix is not None and np.isfinite(max_radius_pix):
        annulus_radii = annulus_radii[annulus_radii <= max_radius_pix]

    for r_pix in annulus_radii:
        CircularAperture((x0, y0), r=r_pix).plot(
            ax=ax,
            color="white",
            lw=0.45,
            alpha=0.35
        )


# ============================================================
# PROFILE PANEL
# ============================================================

def plot_final_profile_axis(ax, galaxy, r25, ruv):
    profiles = {}

    for band in BANDS:
        df = load_profile(galaxy, band)
        if df is None:
            continue

        if band in ["FUV", "NUV"] and np.isfinite(ruv):
            df = df[df["radius_kpc"] <= ruv + UV_CUT_AFTER_RUV_KPC]

        df = cut_after_mu_limit(df, band)

        if len(df) >= 4:
            profiles[band] = df

    for band in BANDS:
        if band not in profiles:
            continue

        df = profiles[band]

        xgrid, ygrid = smooth_profile(
            df["radius_kpc"].values,
            df["mu_AB_arcsec2"].values,
            ngrid=500,
            sigma=1.8,
        )

        if xgrid is None:
            continue

        ax.plot(
            xgrid,
            ygrid,
            color=BAND_COLORS[band],
            lw=2.2,
            label=band,
            solid_capstyle="round",
        )

    if np.isfinite(r25):
        ax.axvline(r25, color="red", ls=":", lw=2.0, zorder=0)
        ax.text(
            r25,
            0.98,
            rf"$R_{{25}}={r25:.1f}$ kpc",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="top",
            fontsize=10,
            color="red",
        )

    if np.isfinite(ruv):
        ax.axvline(ruv, color="tab:blue", ls=":", lw=2.0, zorder=0)
        ax.text(
            ruv,
            0.98,
            rf"$R_{{UV}}={ruv:.1f}$ kpc",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="left",
            va="top",
            fontsize=10,
            color="tab:blue",
        )

    ax.set_xlabel("Radius [kpc]")
    ax.set_ylabel(r"$\mu_{\rm AB}$ [mag arcsec$^{-2}$]")
    ax.invert_yaxis()
    ax.minorticks_on()
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(loc="best", fontsize=9, ncol=2, frameon=True)


# ============================================================
# FINAL GALAXY PANEL
# ============================================================

def make_selected_galaxy_panel(galaxy, r25_dict, ruv_dict):
    fuv_data, fuv_header, fuv_wcs, center, row = load_image_for_display(galaxy, "FUV")
    g_data, g_header, g_wcs, _, _ = load_image_for_display(galaxy, "g")

    if fuv_data is None or g_data is None or center is None or row is None:
        print(f"[SKIP] {galaxy}: missing FUV or g image")
        return None

    r25 = r25_dict.get(galaxy, np.nan)
    ruv = ruv_dict.get(galaxy, np.nan)

    fuv_profile_path = TAB / f"{galaxy}_FUV_radial_profile.csv"
    fuv_profile_full = pd.read_csv(fuv_profile_path) if fuv_profile_path.exists() else None

    base_profile = load_profile(galaxy, "FUV")
    if base_profile is None:
        base_profile = load_profile(galaxy, "g")

    if base_profile is None:
        print(f"[SKIP] {galaxy}: no profile table available")
        return None

    kpc_per_arcsec = get_kpc_per_arcsec(base_profile)

    crop_radius_arcsec = crop_radius_for_galaxy(galaxy, ruv, kpc_per_arcsec)

    fuv_crop, fuv_x1, fuv_y1, fuv_x0_full, fuv_y0_full, fuv_pixscale = crop_image_same_sky_fov(
        fuv_data,
        fuv_wcs,
        center,
        crop_radius_arcsec
    )

    g_crop, g_x1, g_y1, g_x0_full, g_y0_full, g_pixscale = crop_image_same_sky_fov(
        g_data,
        g_wcs,
        center,
        crop_radius_arcsec
    )

    fuv_x0 = fuv_x0_full - fuv_x1
    fuv_y0 = fuv_y0_full - fuv_y1

    g_x0 = g_x0_full - g_x1
    g_y0 = g_y0_full - g_y1

    r25_fuv_pix = radius_kpc_to_pix(r25, kpc_per_arcsec, fuv_pixscale)
    ruv_fuv_pix = radius_kpc_to_pix(ruv, kpc_per_arcsec, fuv_pixscale)

    r25_g_pix = radius_kpc_to_pix(r25, kpc_per_arcsec, g_pixscale)
    ruv_g_pix = radius_kpc_to_pix(ruv, kpc_per_arcsec, g_pixscale)

    fig = plt.figure(figsize=(15.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.35])

    ax_fuv = fig.add_subplot(gs[0, 0])
    ax_g = fig.add_subplot(gs[0, 1])
    ax_prof = fig.add_subplot(gs[0, 2])

    # ------------------------------------------------------------
    # FUV panel
    # ------------------------------------------------------------

    ax_fuv.imshow(
        fuv_crop,
        origin="lower",
        cmap="gray",
        norm=image_norm(fuv_crop)
    )

    # Draw annuli only within the displayed crop radius.
    max_annulus_pix = min(fuv_crop.shape) / 2.0

    overlay_fuv_annuli(
        ax_fuv,
        fuv_x0,
        fuv_y0,
        fuv_profile_full,
        fuv_pixscale,
        max_radius_pix=max_annulus_pix
    )

    overlay_radius(
        ax_fuv,
        fuv_x0,
        fuv_y0,
        r25_fuv_pix,
        r"$R_{25}$",
        "red",
        lw=2.2
    )

    overlay_radius(
        ax_fuv,
        fuv_x0,
        fuv_y0,
        ruv_fuv_pix,
        r"$R_{\rm UV}$",
        "tab:blue",
        lw=2.2
    )

    ax_fuv.scatter(
        [fuv_x0],
        [fuv_y0],
        marker="+",
        s=60,
        c="yellow",
        lw=1.4,
        zorder=40
    )

    lock_image_limits(ax_fuv, fuv_crop)

    ax_fuv.set_title(f"{galaxy}: FUV + annuli", fontsize=13)
    ax_fuv.set_xlabel("Pixel")
    ax_fuv.set_ylabel("Pixel")

    # ------------------------------------------------------------
    # g-band panel
    # ------------------------------------------------------------

    ax_g.imshow(
        g_crop,
        origin="lower",
        cmap="gray",
        norm=image_norm(g_crop)
    )

    overlay_radius(
        ax_g,
        g_x0,
        g_y0,
        r25_g_pix,
        r"$R_{25}$",
        "red",
        lw=2.2
    )

    overlay_radius(
        ax_g,
        g_x0,
        g_y0,
        ruv_g_pix,
        r"$R_{\rm UV}$",
        "tab:blue",
        lw=2.2
    )

    ax_g.scatter(
        [g_x0],
        [g_y0],
        marker="+",
        s=60,
        c="yellow",
        lw=1.4,
        zorder=40
    )

    lock_image_limits(ax_g, g_crop)

    ax_g.set_title(f"{galaxy}: $g$ band", fontsize=13)
    ax_g.set_xlabel("Pixel")
    ax_g.set_ylabel("Pixel")

    # ------------------------------------------------------------
    # Profile panel
    # ------------------------------------------------------------

    plot_final_profile_axis(ax_prof, galaxy, r25, ruv)
    ax_prof.set_title("Final radial profiles", fontsize=13)

    ruv_ratio = (
        ruv / r25
        if np.isfinite(ruv) and np.isfinite(r25) and r25 > 0
        else np.nan
    )

    out_png = FINAL_MANUSCRIPT_DIR / f"{galaxy}_FUV_g_profiles_R25_RUV_sameFOV.png"
    out_pdf = FINAL_MANUSCRIPT_DIR / f"{galaxy}_FUV_g_profiles_R25_RUV_sameFOV.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()

    return {
        "galaxy": galaxy,
        "R25_kpc": r25,
        "RUV_kpc": ruv,
        "RUV_over_R25": ruv_ratio,
        "crop_radius_arcsec": crop_radius_arcsec,
        "output_png": str(out_png),
        "output_pdf": str(out_pdf),
    }


# ============================================================
# LOAD RADII
# ============================================================

r25_dict = load_radius_table(
    TAB / "R25_from_g_r_profiles.csv",
    ["R25_kpc", "R25_g_kpc", "R25_r_kpc", "R25_kpc_from_gr_Bprofile"],
)

ruv_dict = load_radius_table(
    TAB / "UV_radius_from_FUV_sigma_SFR.csv",
    ["R_UV_kpc", "RUV_kpc", "R_UV", "Ruv_kpc"],
)

# Manual corrections in kpc
ruv_dict["NGC5905"] = 57.54 
ruv_dict["NGC5905"] = 50 # RUV = 250 arcsec
# ruv_dict["LEDA135648"] = 75.20


# ============================================================
# RUN
# ============================================================

final_rows = []

for galaxy in SELECTED_FINAL_GALAXIES:
    result = make_selected_galaxy_panel(galaxy, r25_dict, ruv_dict)

    if result is not None:
        final_rows.append(result)
        print(f"[OK] {galaxy}: saved final FUV/g/profile manuscript panel")

final_panel_table = pd.DataFrame(final_rows)

final_panel_table.to_csv(
    TAB / "selected_galaxy_final_panel_outputs_sameFOV.csv",
    index=False
)

print("\nSaved selected-galaxy final panels to:")
print(FINAL_MANUSCRIPT_DIR)

print("\nSaved panel summary table:")
print(TAB / "selected_galaxy_final_panel_outputs_sameFOV.csv")

from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import math

# ============================================================
# COMBINE SELECTED GALAXY PANELS INTO ONE PAGE, TWO COLUMNS
# ============================================================

SELECTED_PANEL_PDF = FIG / "selected_galaxy_FUV_g_profiles_R25_RUV_panels_onepage_2cols.pdf"
SELECTED_PANEL_PNG = FIG / "selected_galaxy_FUV_g_profiles_R25_RUV_panels_onepage_2cols.png"

# Collect panels in selected-galaxy order
selected_imgs = []

for galaxy in SELECTED_FINAL_GALAXIES:
    f1 = FINAL_MANUSCRIPT_DIR / f"{galaxy}_FUV_g_profiles_R25_RUV_sameFOV.png"
    f2 = FINAL_MANUSCRIPT_DIR / f"{galaxy}_FUV_g_profiles_R25_RUV.png"

    if f1.exists():
        selected_imgs.append(f1)
    elif f2.exists():
        selected_imgs.append(f2)

if len(selected_imgs) == 0:
    raise FileNotFoundError(f"No selected-galaxy panel PNG files found in {FINAL_MANUSCRIPT_DIR}")

print(f"Found {len(selected_imgs)} selected-galaxy panels.")

# ============================================================
# PAGE SETTINGS
# ============================================================

ncols = 2
nrows = math.ceil(len(selected_imgs) / ncols)

# Large manuscript-checking canvas
# Increase page_h if panels look too compressed vertically.
page_w = 3600
page_h = 5200

outer_pad = 70
col_gap = 45
row_gap = 35
title_h = 80

usable_w = page_w - 2 * outer_pad - col_gap
usable_h = page_h - 2 * outer_pad - title_h - (nrows - 1) * row_gap

cell_w = usable_w // ncols
cell_h = usable_h // nrows

page = Image.new("RGB", (page_w, page_h), "white")
draw = ImageDraw.Draw(page)

title = "Selected GLSB FUV, g-band, and radial-profile panels"
draw.text((outer_pad, 35), title, fill="black")

# ============================================================
# PLACE IMAGES
# ============================================================

for i, f in enumerate(selected_imgs):

    row = i // ncols
    col = i % ncols

    x_cell = outer_pad + col * (cell_w + col_gap)
    y_cell = outer_pad + title_h + row * (cell_h + row_gap)

    im = Image.open(f).convert("RGB")

    # Fit panel into each cell
    im = ImageOps.contain(
        im,
        (cell_w, cell_h),
        method=Image.Resampling.LANCZOS
    )

    x = x_cell + (cell_w - im.width) // 2
    y = y_cell + (cell_h - im.height) // 2

    page.paste(im, (x, y))

# ============================================================
# SAVE
# ============================================================

page.save(SELECTED_PANEL_PNG, dpi=(300, 300))
page.save(SELECTED_PANEL_PDF, "PDF", resolution=300)

print(f"[SAVED PNG] {SELECTED_PANEL_PNG}")
print(f"[SAVED PDF] {SELECTED_PANEL_PDF}")
print(f"Panels combined: {len(selected_imgs)}")
print(f"Layout: {nrows} rows × {ncols} columns")



