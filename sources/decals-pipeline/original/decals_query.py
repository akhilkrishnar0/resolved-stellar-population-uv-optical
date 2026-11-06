#updated code with all files from decals, maximum size 3000 pixels
import os
import shutil
import requests
from astroquery.query import BaseQuery
from astropy.coordinates import SkyCoord

class DECaLSQuery(BaseQuery):
    URL_TEMPLATE_JPG = 'https://www.legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}&pix=0.25&layer={layer}&size={size}'
    URL_TEMPLATE_FITS = 'https://www.legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&pix=0.25&layer={layer}&size={size}'
    URL_TEMPLATE_PSF = 'https://www.legacysurvey.org/viewer/coadd-psf/?ra={ra}&dec={dec}&layer={layer}'
    URL_TEMPLATE_INVVAR = 'https://www.legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&layer={layer}&size={size}&subimage'

    def __init__(self, output_dir="downloads", layer="ls-dr9"):
        self.output_dir = output_dir
        self.layer = layer

        os.makedirs(output_dir, exist_ok=True)
        self.image_dir = os.path.join(output_dir, "images")
        self.fits_dir = os.path.join(output_dir, "fits")
        self.psf_dir = os.path.join(output_dir, "psf")
        self.invvar_dir = os.path.join(output_dir, "invvar")

        for d in [self.image_dir, self.fits_dir, self.psf_dir, self.invvar_dir]:
            os.makedirs(d, exist_ok=True)

    def _get_cutout_url(self, ra, dec, size, filetype='jpg'):
        if filetype == 'jpg':
            return self.URL_TEMPLATE_JPG.format(ra=ra, dec=dec, size=size, layer=self.layer)
        elif filetype == 'fits':
            return self.URL_TEMPLATE_FITS.format(ra=ra, dec=dec, size=size, layer=self.layer)
        elif filetype == 'invvar':
            return self.URL_TEMPLATE_INVVAR.format(ra=ra, dec=dec, size=size, layer=self.layer)
        elif filetype == 'psf':
            return self.URL_TEMPLATE_PSF.format(ra=ra, dec=dec, layer=self.layer)
        else:
            raise ValueError(f"Unsupported file type: {filetype}")

    def download_image(self, galid, ra, dec, size):
        url = self._get_cutout_url(ra, dec, size, 'jpg')
        self._download_file(url, os.path.join(self.image_dir, f"{galid}.jpg"))

    def download_fits(self, galid, ra, dec, size):
        url = self._get_cutout_url(ra, dec, size, 'fits')
        self._download_file(url, os.path.join(self.fits_dir, f"{galid}.fits"))

    def download_psf(self, galid, ra, dec):
        url = self._get_cutout_url(ra, dec, size=None, filetype='psf')
        self._download_file(url, os.path.join(self.psf_dir, f"{galid}_psf.fits"))

    def download_invvar(self, galid, ra, dec, size):
        url = self._get_cutout_url(ra, dec, size, 'invvar')
        self._download_file(url, os.path.join(self.invvar_dir, f"{galid}_invvar.fits"))

    def _download_file(self, url, file_name):
        print(f"Downloading from {url}")
        res = requests.get(url, stream=True)
        if res.status_code == 200:
            with open(file_name, 'wb') as f:
                shutil.copyfileobj(res.raw, f)
            print(f"Saved: {file_name}")
        else:
            print(f"Failed: {url}")

    def query_region(self, coordinates, size, download_type='all', galid=None):
        ra = coordinates.ra.deg
        dec = coordinates.dec.deg
        galid = galid or f"decals_{ra}_{dec}"

        if download_type in ('jpg', 'all'):
            self.download_image(galid, ra, dec, size)

        if download_type in ('fits', 'all'):
            self.download_fits(galid, ra, dec, size)

        if download_type in ('psf', 'all'):
            self.download_psf(galid, ra, dec)

        if download_type in ('invvar', 'all'):
            self.download_invvar(galid, ra, dec, size)
