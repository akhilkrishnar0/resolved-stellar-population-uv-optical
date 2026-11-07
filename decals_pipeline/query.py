#updated code with all files from decals, maximum size 3000 pixels
import os
import time
import json
from pathlib import Path
import requests
from astroquery.query import BaseQuery
from astropy.coordinates import SkyCoord

class DECaLSQuery(BaseQuery):
    URL_TEMPLATE_JPG = 'https://www.legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}&pixscale=0.262&layer={layer}&bands=grz&size={size}'
    URL_TEMPLATE_FITS = 'https://www.legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&pixscale=0.262&layer={layer}&bands=grz&size={size}'
    URL_TEMPLATE_PSF = 'https://www.legacysurvey.org/viewer/coadd-psf/?ra={ra}&dec={dec}&layer={layer}&bands=grz'
    URL_TEMPLATE_INVVAR = 'https://www.legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&layer={layer}&bands=grz&size={size}&subimage'

    def __init__(self, output_dir="downloads", layer="ls-dr9", overwrite=True):
        self.output_dir = output_dir
        self.layer = layer
        self.overwrite = overwrite

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

    @staticmethod
    def _valid_file(path):
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if path.suffix == '.fits':
            from astropy.io import fits
            try:
                with fits.open(path) as hdul:
                    hdul.verify('exception')
                    if not any(h.header.get('NAXIS', 0) >= 2 for h in hdul):
                        return False
            except (OSError, ValueError):
                return False
        return True

    def _download_file(self, url, file_name):
        target = Path(file_name)
        metadata = target.with_name(target.name + '.request.json')
        same_request = True
        if metadata.exists():
            try:
                same_request = json.loads(metadata.read_text()).get('url') == url
            except (OSError, ValueError):
                same_request = False
        if not self.overwrite and same_request and self._valid_file(target):
            print(f"Using existing: {target}")
            if not metadata.exists():
                print("Existing file has no request metadata; confirm its centre, size and layer match.")
            return
        temporary = target.with_name(target.name + '.part')
        transient = {408, 429, 500, 502, 503, 504}
        for attempt in range(1, 4):
            delay = 2**attempt
            try:
                print(f"Downloading ({attempt}/3): {url}")
                with requests.get(url, stream=True, timeout=(30, 300)) as response:
                    retry_after = response.headers.get('Retry-After', '')
                    if retry_after.isdigit():
                        delay = min(60, max(delay, int(retry_after)))
                    response.raise_for_status()
                    with temporary.open('wb') as output:
                        for chunk in response.iter_content(chunk_size=1024*1024):
                            if chunk:
                                output.write(chunk)
                if not temporary.stat().st_size:
                    raise OSError(f"Empty download: {url}")
                if target.suffix == '.fits':
                    from astropy.io import fits
                    with fits.open(temporary) as hdul:
                        hdul.verify('exception')
                temporary.replace(target)
                metadata.write_text(json.dumps({'url':url}, indent=2)+'\n')
                print(f"Saved: {target}")
                return
            except requests.RequestException as exc:
                status = exc.response.status_code if exc.response is not None else None
                retryable = status in transient or isinstance(exc, (
                    requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError))
                if not retryable:
                    raise
                if attempt == 3:
                    raise RuntimeError(f"DECaLS download failed after 3 attempts: {url}. "
                        "Completed files are retained. Retry with download='missing'. "
                        "The requested size, pixel scale and layer were not changed.") from exc
                print(f"Temporary download failure ({status or type(exc).__name__}); retrying in {delay}s")
            finally:
                temporary.unlink(missing_ok=True)
            time.sleep(delay)

    def query_region(self, coordinates, size, download_type='all', galid=None):
        if download_type not in ('jpg', 'fits', 'psf', 'invvar', 'all'):
            raise ValueError(f"Unsupported download type: {download_type}")
        if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= 3000:
            raise ValueError("size must be an integer from 1 to 3000 pixels (package limit)")
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

