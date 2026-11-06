import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u



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
        if result is None or len(result) == 0:
            raise ValueError(f"{target} not found in SIMBAD")

        if 'ra' in result.colnames and 'dec' in result.colnames:
            return SkyCoord(result['ra'][0], result['dec'][0], unit=u.deg)
        return SkyCoord(result['RA'][0] + " " + result['DEC'][0],
                        unit=(u.hourangle, u.deg))

    # ---------- CHECK ----------
    def already_downloaded(self, target):
        g = self.clean_name(target)
        f = self.download_dir / "fits" / f"{g}.fits"
        return f.exists()

    # ---------- DOWNLOAD ----------
    def download(self, target, size=1200):
        from .query import DECaLSQuery
        g = self.clean_name(target)

        coord = self.get_coord(target)
        query = DECaLSQuery(output_dir=str(self.download_dir), layer=self.layer)

        print(f"⬇ Downloading {g}")
        query.query_region(coord, size=size, download_type='all', galid=g)

    def _check_image_invvar(self, target):
        name = self.clean_name(target)
        with fits.open(self.download_dir / 'fits' / f'{name}.fits') as images, fits.open(self.download_dir / 'invvar' / f'{name}_invvar.fits') as inv:
            cube = images[0].data
            if cube is None or cube.ndim != 3 or cube.shape[0] != 3:
                raise ValueError('Expected primary image cube (3, ny, nx), ordered g/r/z')
            if len(inv) != 7:
                raise ValueError('Expected one image/INVVAR pair per g/r/z band (7 HDUs). Multiple-brick subimages require explicit selection or mosaicking.')
            wcs = WCS(images[0].header).celestial
            yy, xx = np.meshgrid(np.linspace(0, cube.shape[1]-1, 3), np.linspace(0, cube.shape[2]-1, 3), indexing='ij')
            for band, index in zip(('g', 'r', 'z'), (2, 4, 6)):
                hdu = inv[index]
                label = str(hdu.header.get('BAND', hdu.header.get('FILTER', band))).strip().lower()
                if label != band:
                    raise ValueError(f'Unexpected band in inverse-variance HDU {index}: {label}')
                if hdu.data is None or hdu.data.shape != cube.shape[1:]:
                    raise ValueError(f'{band} image/inverse-variance shapes differ; native subimages cannot be used as aligned resampled maps')
                ix, iy = WCS(hdu.header).celestial.world_to_pixel(wcs.pixel_to_world(xx, yy))
                if not (np.allclose(ix, xx, rtol=0, atol=1e-3) and np.allclose(iy, yy, rtol=0, atol=1e-3)):
                    raise ValueError(f'{band} image/inverse-variance WCS mismatch. Use matched native products or a documented variance-resampling procedure.')

    # ---------- LOAD ----------
    def load_data(self, target):
        self._check_image_invvar(target)
        g = self.clean_name(target)

        fits_file   = self.download_dir / "fits" / f"{g}.fits"
        invvar_file = self.download_dir / "invvar" / f"{g}_invvar.fits"
        psf_file    = self.download_dir / "psf" / f"{g}_psf.fits"

        if not fits_file.exists():
            raise FileNotFoundError(f"{fits_file} not found")

        with fits.open(fits_file) as hdul:
            img = hdul[0].data
            if img.ndim != 3 or img.shape[0] != 3:
                raise ValueError("Expected a primary image cube shaped (3, ny, nx), ordered g/r/z")
            images = [img[i].copy() for i in range(3)]
            wcs = WCS(hdul[0].header).celestial
        with fits.open(invvar_file) as hdul:
            invvars = [hdul[i].data.copy() for i in (2, 4, 6)]
        if any(a.shape != b.shape for a, b in zip(images, invvars)):
            raise ValueError("Image and inverse-variance shapes differ")
        psfs, _ = self._read_psfs(psf_file)

        # SIGMA
        sigmas = []
        for inv in invvars:
            s = np.full(inv.shape, np.nan, dtype=float)
            m = np.isfinite(inv) & (inv > 0)
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

        psfs, headers = self._read_psfs(psf_file)
        for band, data, header in zip(bands, psfs, headers):
            header["BAND"] = band
            fits.writeto(self.download_dir / "psf" / f"{g}_PSF_{band}.fits",
                         data, header=header, overwrite=True)

        print(f"✔ Split PSF for {g}")

    @staticmethod
    def _read_psfs(path):
        # Retain the uploaded split_psf convention: one 2-D HDU per g/r/z band.
        # Reject ambiguous cubes rather than guessing the band axis.
        with fits.open(path) as hdul:
            planes = [h for h in hdul if h.data is not None and h.data.ndim == 2]
            if len(planes) != 3:
                raise ValueError("Expected three 2-D PSF HDUs in g/r/z order; inspect the FITS headers to identify other layouts")
            return [h.data.copy() for h in planes], [h.header.copy() for h in planes]

    @staticmethod
    def _image_header(header, band):
        result = WCS(header).celestial.to_header()
        for key in ('BUNIT', 'TELESCOP', 'INSTRUME', 'SURVEY', 'MAGZERO'):
            if key in header:
                result[key] = header[key]
        result['BAND'] = band
        result.add_history('Split from original DECaLS product without resampling.')
        return result

    def save_split_data(self, target):
        self._check_image_invvar(target)
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
            if img.ndim != 3 or img.shape[0] != 3:
                raise ValueError("Expected image cube (3, ny, nx) in g/r/z order")
    
            for i, band in enumerate(bands):
                data = img[i,:,:]
    
                h = self._image_header(header, band)
    
                out = out_img / f"{g}_{band}.fits"
                fits.writeto(out, data, header=h, overwrite=True)
    
        # ---------- INVVAR ----------
        with fits.open(invvar_file) as hdul:
            invs = [hdul[2].data, hdul[4].data, hdul[6].data]
    
            for i, band in enumerate(bands):
                data = invs[i]
    
                h = self._image_header(hdul[(2, 4, 6)[i]].header, band)
    
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

            if do_plot:
                self.plot(t)
            if do_save_split:
                self.save_split_data(t)
            if do_psf_split:
                self.split_psf(t)
