from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization import make_lupton_rgb

from astroquery.simbad import Simbad


# ============================================================
# SIMBAD CENTER RESOLVER
# ============================================================

def get_simbad_centers(galaxies):
    """
    Resolve galaxy centers from SIMBAD.

    Returns
    -------
    pandas.DataFrame
        Columns: galaxy, ra_simbad, dec_simbad, ra_deg, dec_deg
    """

    simbad = Simbad()
    simbad.add_votable_fields("ra", "dec")

    rows = []

    for name in galaxies:
        try:
            result = simbad.query_object(name)

            if result is None:
                print(f"❌ SIMBAD could not resolve: {name}")
                rows.append({
                    "galaxy": name,
                    "ra_simbad": None,
                    "dec_simbad": None,
                    "ra_deg": np.nan,
                    "dec_deg": np.nan
                })
                continue

            ra = result["RA"][0]
            dec = result["DEC"][0]

            coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")

            rows.append({
                "galaxy": name,
                "ra_simbad": ra,
                "dec_simbad": dec,
                "ra_deg": coord.ra.deg,
                "dec_deg": coord.dec.deg
            })

            print(f"✔ {name}: RA={coord.ra.deg:.6f}, Dec={coord.dec.deg:.6f}")

        except Exception as e:
            print(f"❌ SIMBAD failed for {name}: {e}")
            rows.append({
                "galaxy": name,
                "ra_simbad": None,
                "dec_simbad": None,
                "ra_deg": np.nan,
                "dec_deg": np.nan
            })

    return pd.DataFrame(rows)


# ============================================================
# MULTIBAND CUTOUT PIPELINE
# ============================================================

class MultiBandCutoutPipeline:

    def __init__(
        self,
        galex_dir="downloaded_galex",
        decals_dir="decals_data/outputs/images",
        out_dir="cutouts"
    ):

        self.galex_dir = Path(galex_dir).resolve()
        self.decals_dir = Path(decals_dir).resolve()
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def clean_name(self, g):
        return g.replace(" ", "")

    # --------------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------------

    def get_files(self, galaxy):
        g = self.clean_name(galaxy)

        return {
            "FUV": self.galex_dir / g / "cropped" / "FUV_stacked.fits",
            "NUV": self.galex_dir / g / "cropped" / "NUV_stacked.fits",
            "g":   self.decals_dir / f"{g}_g.fits",
            "r":   self.decals_dir / f"{g}_r.fits",
            "z":   self.decals_dir / f"{g}_z.fits",
        }

    # --------------------------------------------------------
    # CROP
    # --------------------------------------------------------

    # def crop(self, fits_file, center, size_arcsec):
    #     data, header = fits.getdata(fits_file, header=True)
    #     wcs = WCS(header)

    #     cut = Cutout2D(
    #         data,
    #         position=center,
    #         size=(size_arcsec * u.arcsec, size_arcsec * u.arcsec),
    #         wcs=wcs,
    #         mode="partial",
    #         fill_value=np.nan
    #     )

    #     return cut.data, cut.wcs.to_header()

    def crop(self, fits_file, center, size_arcsec):
        data, header = fits.getdata(fits_file, header=True)
        wcs = WCS(header)
    
        cut = Cutout2D(
            data,
            position=center,
            size=(size_arcsec * u.arcsec, size_arcsec * u.arcsec),
            wcs=wcs,
            mode="partial",
            fill_value=np.nan
        )
    
        # --------------------------------------------------------
        # Keep the full original header
        # --------------------------------------------------------
        new_header = header.copy()
    
        # --------------------------------------------------------
        # Update only the WCS keywords using the cutout WCS
        # --------------------------------------------------------
        new_wcs_header = cut.wcs.to_header()
    
        # Remove old WCS-related keywords from original header
        wcs_keys_to_remove = [
            "WCSAXES",
            "CRPIX1", "CRPIX2",
            "CRVAL1", "CRVAL2",
            "CTYPE1", "CTYPE2",
            "CUNIT1", "CUNIT2",
            "CDELT1", "CDELT2",
            "CD1_1", "CD1_2", "CD2_1", "CD2_2",
            "PC1_1", "PC1_2", "PC2_1", "PC2_2",
            "CROTA1", "CROTA2",
            "LONPOLE", "LATPOLE",
            "RADESYS", "EQUINOX",
            "PV1_0", "PV1_1", "PV1_2", "PV1_3",
            "PV2_0", "PV2_1", "PV2_2", "PV2_3",
            "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER",
        ]
    
        for key in wcs_keys_to_remove:
            if key in new_header:
                del new_header[key]
    
        # Add updated cutout WCS
        new_header.update(new_wcs_header)
    
        # --------------------------------------------------------
        # Update image size
        # --------------------------------------------------------
        new_header["NAXIS"] = 2
        new_header["NAXIS1"] = cut.data.shape[1]
        new_header["NAXIS2"] = cut.data.shape[0]
    
        # --------------------------------------------------------
        # Add useful processing information
        # --------------------------------------------------------
        new_header["HISTORY"] = "Cutout made with astropy.nddata.Cutout2D"
        new_header["HISTORY"] = "Original header preserved"
        new_header["HISTORY"] = "WCS updated to cutout WCS"
    
        return cut.data, new_header
    

    # --------------------------------------------------------
    # PROCESS ONE GALAXY
    # --------------------------------------------------------

    def process_galaxy(self, galaxy, ra_deg, dec_deg, size_arcsec=60):
        g = self.clean_name(galaxy)
        print(f"\n=== Cropping {g} ===")

        files = self.get_files(galaxy)

        out_gal = self.out_dir / g
        out_gal.mkdir(parents=True, exist_ok=True)

        center = SkyCoord(ra_deg, dec_deg, unit="deg", frame="icrs")

        for band, f in files.items():
            if not f.exists():
                print(f"⚠ Missing {band}: {f}")
                continue

            try:
                data, hdr = self.crop(f, center, size_arcsec)
                fits.writeto(out_gal / f"{band}.fits", data, hdr, overwrite=True)
                print(f"✔ {band}: saved {data.shape}")

            except Exception as e:
                print(f"❌ {band}: {e}")

    # --------------------------------------------------------
    # SCALE BAR
    # --------------------------------------------------------

    def get_scale_bar(self, distance_mpc):
        distance_mpc = float(distance_mpc)

        theta_rad = 10.0 / (distance_mpc * 1000.0)
        arcsec = theta_rad * 206265.0

        pixscale = {
            "FUV": 1.5,
            "NUV": 1.5,
            "g": 0.262,
            "r": 0.262,
            "z": 0.262
        }

        return {b: arcsec / pixscale[b] for b in pixscale}

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    def plot(self, galaxy, distance_mpc=10):
        g = self.clean_name(galaxy)
        gal_dir = self.out_dir / g

        bands = ["FUV", "NUV", "g", "r", "z"]
        data = {}
        wcs_dict = {}

        for b in bands:
            f = gal_dir / f"{b}.fits"

            if not f.exists():
                print(f"❌ Missing {b}: {f}")
                return

            d, h = fits.getdata(f, header=True)
            data[b] = d
            wcs_dict[b] = WCS(h)

        bars = self.get_scale_bar(distance_mpc)

        # ----------------------------------------------------
        # SINGLE-BAND ROW PLOT
        # ----------------------------------------------------

        fig = plt.figure(figsize=(18, 4))

        for i, b in enumerate(bands):
            ax = plt.subplot(1, 5, i + 1, projection=wcs_dict[b])

            d = data[b]
            finite = np.isfinite(d)

            if finite.sum() == 0:
                print(f"⚠ No finite data in {b}")
                continue

            vmin, vmax = np.nanpercentile(d[finite], [5, 99])

            ax.imshow(
                d,
                origin="lower",
                cmap="gray",
                vmin=vmin,
                vmax=vmax
            )

            ax.set_title(b, fontsize=12)
            ax.set_xlabel("RA")
            ax.set_ylabel("Dec")
            ax.coords.grid(color="white", ls="dotted", alpha=0.4)

            ny, nx = d.shape
            bar = bars[b]

            x0 = nx * 0.10
            y0 = ny * 0.10

            ax.plot([x0, x0 + bar], [y0, y0], color="white", lw=3)
            ax.text(x0, y0 + ny * 0.05, "10 kpc", color="white", fontsize=9)

        plt.tight_layout()
        plt.savefig(gal_dir / f"{g}_multiband.png", dpi=200, bbox_inches="tight")
        plt.close()

        print(f"✔ Saved multiband plot: {gal_dir / f'{g}_multiband.png'}")

        # ----------------------------------------------------
        # RGB PLOTS
        # ----------------------------------------------------

        uv_rgb = make_lupton_rgb(
            data["NUV"],
            data["NUV"],
            data["FUV"] * 5,
            stretch=0.05,
            Q=10
        )

        opt_rgb = make_lupton_rgb(
            data["z"],
            data["r"],
            data["g"],
            stretch=0.1,
            Q=10
        )

        fig = plt.figure(figsize=(10, 5))

        # UV RGB
        ax1 = plt.subplot(1, 2, 1, projection=wcs_dict["FUV"])
        ax1.imshow(uv_rgb, origin="lower")
        ax1.set_title("FUV-NUV RGB")
        ax1.coords.grid(color="white", ls="dotted", alpha=0.4)
        ax1.set_xlabel("RA")
        ax1.set_ylabel("Dec")

        ny, nx = data["FUV"].shape
        bar_uv = bars["FUV"]
        x0 = nx * 0.10
        y0 = ny * 0.10

        ax1.plot([x0, x0 + bar_uv], [y0, y0], color="white", lw=3)
        ax1.text(x0, y0 + ny * 0.05, "10 kpc", color="white", fontsize=9)

        # Optical RGB
        ax2 = plt.subplot(1, 2, 2, projection=wcs_dict["g"])
        ax2.imshow(opt_rgb, origin="lower")
        ax2.set_title("g-r-z RGB")
        ax2.coords.grid(color="white", ls="dotted", alpha=0.4)
        ax2.set_xlabel("RA")
        ax2.set_ylabel("Dec")

        ny, nx = data["g"].shape
        bar_opt = bars["g"]
        x0 = nx * 0.10
        y0 = ny * 0.10

        ax2.plot([x0, x0 + bar_opt], [y0, y0], color="white", lw=3)
        ax2.text(x0, y0 + ny * 0.05, "10 kpc", color="white", fontsize=9)

        plt.tight_layout()
        plt.savefig(gal_dir / f"{g}_RGB.png", dpi=200, bbox_inches="tight")
        plt.close()

        print(f"✔ Saved RGB plot: {gal_dir / f'{g}_RGB.png'}")

    # --------------------------------------------------------
    # CHECK CUTOUT CENTERS
    # --------------------------------------------------------

    def check_cutout_centers(self, galaxy):
        g = self.clean_name(galaxy)
        gal_dir = self.out_dir / g

        print(f"\n=== Checking cutout centers for {g} ===")

        for b in ["FUV", "NUV", "g", "r", "z"]:
            f = gal_dir / f"{b}.fits"

            if not f.exists():
                print(f"⚠ Missing {b}")
                continue

            data, header = fits.getdata(f, header=True)
            wcs = WCS(header)

            ny, nx = data.shape
            center = wcs.pixel_to_world((nx - 1) / 2, (ny - 1) / 2)

            print(
                f"{b:3s}: "
                f"RA={center.ra.deg:.6f}, "
                f"Dec={center.dec.deg:.6f}, "
                f"shape={data.shape}"
            )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(self, galaxies, ras, decs, distances, sizes=60, do_plot=True):

        if isinstance(galaxies, str):
            galaxies = [galaxies]

        if isinstance(sizes, (int, float)):
            sizes = [sizes] * len(galaxies)

        if not (
            len(galaxies)
            == len(ras)
            == len(decs)
            == len(distances)
            == len(sizes)
        ):
            raise ValueError(
                "galaxies, ras, decs, distances, and sizes must have same length"
            )

        for g, ra, dec, d, s in zip(galaxies, ras, decs, distances, sizes):
            print(
                f"\n=== {g} "
                f"(RA={ra:.6f}, Dec={dec:.6f}, D={d} Mpc, size={s} arcsec) ==="
            )

            self.process_galaxy(
                g,
                ra_deg=ra,
                dec_deg=dec,
                size_arcsec=s
            )

            if do_plot:
                try:
                    self.plot(g, distance_mpc=float(d))
                except Exception as e:
                    print(f"❌ Plot failed for {g}: {e}")