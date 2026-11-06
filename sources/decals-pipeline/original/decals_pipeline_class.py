import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from decals_query import DECaLSQuery


class DecalsPipeline:
    def __init__(self, base_dir="decals_data", layer="ls-dr9"):
        self.base_dir = Path(base_dir).resolve()
        self.layer = layer

        # directories
        self.download_dir = self.base_dir / "downloads"
        self.plot_dir = self.base_dir / "plots"

        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def clean_name(self, g):
        return g.replace(" ", "")

    # ---------- COORD ----------
    def get_coord(self, target):
        result = Simbad.query_object(target)
        if result is None:
            raise ValueError(f"{target} not found in SIMBAD")

        return SkyCoord(result['RA'][0] + " " + result['DEC'][0],
                        unit=(u.hourangle, u.deg))

    # ---------- CHECK ----------
    def already_downloaded(self, target):
        g = self.clean_name(target)
        f = self.download_dir / "fits" / f"{g}.fits"
        return f.exists()

    # ---------- DOWNLOAD ----------
    def download(self, target, size=1200):
        g = self.clean_name(target)

        coord = self.get_coord(target)
        query = DECaLSQuery(output_dir=str(self.download_dir), layer=self.layer)

        print(f"⬇ Downloading {g}")
        query.query_region(coord, size=size, download_type='all', galid=g)

    # ---------- LOAD ----------
    def load_data(self, target):
        g = self.clean_name(target)

        fits_file   = self.download_dir / "fits" / f"{g}.fits"
        invvar_file = self.download_dir / "invvar" / f"{g}_invvar.fits"
        psf_file    = self.download_dir / "psf" / f"{g}_psf.fits"

        if not fits_file.exists():
            raise FileNotFoundError(f"{fits_file} not found")

        # IMAGE
        hdul = fits.open(fits_file)
        img = hdul[0].data
        wcs = WCS(hdul[0].header, naxis=2)
        images = [img[i,:,:] for i in range(3)]

        # INVVAR
        inv_hdul = fits.open(invvar_file)
        invvars = [inv_hdul[2].data, inv_hdul[4].data, inv_hdul[6].data]

        # PSF
        psf = fits.getdata(psf_file)
        psfs = [psf[:,:,i] if psf.ndim==3 else psf for i in range(3)]

        # SIGMA
        sigmas = []
        for inv in invvars:
            s = np.zeros_like(inv)
            m = inv > 0
            s[m] = 1.0 / np.sqrt(inv[m])
            sigmas.append(s)

        return images, invvars, sigmas, psfs, wcs

    # ---------- PLOT ----------
    def plot(self, target):
        g = self.clean_name(target)
        bands = ['g','r','z']

        images, invvars, sigmas, psfs, wcs = self.load_data(target)

        fig = plt.figure(figsize=(12,16))

        for row in range(4):
            for col in range(3):
                ax = plt.subplot(4,3,row*3+col+1,
                                 projection=wcs if row < 3 else None)

                if row == 0:
                    data = images[col]
                    vmin,vmax = np.percentile(data[np.isfinite(data)],[21,97])
                    ax.imshow(data, vmin=vmin, vmax=vmax)
                    ax.set_title(f"{bands[col]} image")

                elif row == 1:
                    data = invvars[col]
                    vmin,vmax = np.percentile(data[np.isfinite(data)],[1,99])
                    ax.imshow(data, vmin=vmin, vmax=vmax)
                    ax.set_title(f"{bands[col]} invvar")

                elif row == 2:
                    data = sigmas[col]
                    vmin,vmax = np.percentile(data[np.isfinite(data)],[1,99])
                    ax.imshow(data, vmin=vmin, vmax=vmax)
                    ax.set_title(f"{bands[col]} sigma")

                else:
                    ax.imshow(psfs[col])
                    ax.set_title(f"{bands[col]} PSF")
                    ax.axis("off")
                    continue

                ax.coords.grid(color='white', ls='dotted', alpha=0.5)
                ax.set_xlabel("RA")
                ax.set_ylabel("Dec")

        out = self.plot_dir / f"{g}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()

        print(f"✔ Saved plot {out}")

    # ---------- PSF SPLIT ----------
    def split_psf(self, target):
        g = self.clean_name(target)
        bands = ["g","r","z"]

        psf_file = self.download_dir / "psf" / f"{g}_psf.fits"

        with fits.open(psf_file) as hdul:
            for i, band in enumerate(bands):
                data = hdul[i].data
                header = hdul[i].header

                out = self.download_dir / "psf" / f"{g}_PSF_{band}.fits"

                h = header.copy()
                h["BAND"] = band

                fits.writeto(out, data, header=h, overwrite=True)

        print(f"✔ Split PSF for {g}")

    def save_split_data(self, target):
        g = self.clean_name(target)
        bands = ["g","r","z"]
    
        # create output dirs
        out_img = self.base_dir / "outputs" / "images"
        out_inv = self.base_dir / "outputs" / "invvar"
        out_img.mkdir(parents=True, exist_ok=True)
        out_inv.mkdir(parents=True, exist_ok=True)
    
        fits_file   = self.download_dir / "fits" / f"{g}.fits"
        invvar_file = self.download_dir / "invvar" / f"{g}_invvar.fits"
    
        # ---------- IMAGE ----------
        with fits.open(fits_file) as hdul:
            img = hdul[0].data
            header = hdul[0].header
    
            for i, band in enumerate(bands):
                data = img[i,:,:]
    
                h = header.copy()
                h["BAND"] = band
    
                out = out_img / f"{g}_{band}.fits"
                fits.writeto(out, data, header=h, overwrite=True)
    
        # ---------- INVVAR ----------
        with fits.open(invvar_file) as hdul:
            invs = [hdul[2].data, hdul[4].data, hdul[6].data]
    
            for i, band in enumerate(bands):
                data = invs[i]
    
                h = hdul[2].header.copy()  # base header
                h["BAND"] = band
    
                out = out_inv / f"{g}_{band}_invvar.fits"
                fits.writeto(out, data, header=h, overwrite=True)
    
        print(f"✔ Saved split IMAGE + INVVAR for {g}")

    

    # ---------- RUN ----------
    def run(self, targets, size=1200,
            skip_download=True,
            force_download=False,
            do_plot=True,
            do_psf_split=True,
            do_save_split=True):

        if isinstance(targets, str):
            targets = [targets]

        for t in targets:
            print(f"\n=== {t} ===")

            # DOWNLOAD CONTROL (fixed)
            if force_download:
                print(f"🔁 Force downloading {t}")
                self.download(t, size=size)

            elif skip_download and self.already_downloaded(t):
                print(f"⏭ Skipping download for {t}")

            else:
                self.download(t, size=size)

            # PLOT
            if do_plot:
                try:
                    self.plot(t)
                except Exception as e:
                    print(f"❌ Plot failed: {e}")
            # SAVE SPLIT DATA
            if do_save_split:
                try:
                    self.save_split_data(t)
                except Exception as e:
                    print(f"❌ Save split failed: {e}")

            # PSF SPLIT
            if do_psf_split:
                try:
                    self.split_psf(t)
                except Exception as e:
                    print(f"❌ PSF split failed: {e}")