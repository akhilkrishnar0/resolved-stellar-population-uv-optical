"""Notebook-facing orchestration with explicit, unconditional skip flags."""
from pathlib import Path
from astropy import units as u
from astropy.io import fits
from .download import download_galex
from .pipeline import process_galaxy
from .plotter import plot_fuv_nuv
from .io_utils import image_pairs


class GalexPipeline:
    def __init__(self, base_dir='downloaded_galex', plot_dir='galex_plots', retries=3):
        self.base_dir = Path(base_dir).resolve()
        self.plot_dir = Path(plot_dir).resolve()
        self.retries = retries
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def clean_name(g):
        return g.replace(' ', '')

    def is_done(self, galaxy):
        """Check new-format outputs; does not certify inputs or crop size are unchanged."""
        folder = self.base_dir / self.clean_name(galaxy) / 'cropped'
        for band in ('FUV', 'NUV'):
            for suffix in ('stacked', 'rrhr', 'coverage'):
                path = folder / f'{band}_{suffix}.fits'
                if not path.is_file() or fits.getheader(path).get('WGHTTYPE') != 'RRHR':
                    return False
        return True

    def already_downloaded(self, galaxy):
        """Local paired-data check only, NOT archive completeness."""
        folder = self.base_dir / self.clean_name(galaxy)
        try:
            pairs = [image_pairs(folder, band) for band in ('FUV', 'NUV')]
        except FileNotFoundError:
            return False
        return any(pairs)

    def download_galex(self, galaxy, size=0.1*u.deg, unzip=True):
        if not unzip:
            raise ValueError('This pipeline requires extracted FITS pairs; use unzip=True')
        return download_galex(galaxy, root=self.base_dir, radius=size, retries=self.retries)

    def run_download(self, galaxy):
        return self.download_galex(galaxy)

    def process(self, galaxy, crop_size=300):
        # Bare numbers retain the original wrapper's pixel-size convention.
        return process_galaxy(self.clean_name(galaxy), root=self.base_dir, crop_size=crop_size)

    def plot(self, galaxy):
        return plot_fuv_nuv(galaxy, self.base_dir, self.plot_dir)

    def run(self, galaxies, crop_size=300, skip_download=True,
            skip_process=False, skip_plot=False):
        """True ALWAYS skips that stage. False runs it; failures propagate."""
        results = {}
        for galaxy in galaxies:
            result = {}
            if not skip_download:
                result['download'] = self.run_download(galaxy)
            if not skip_process:
                result['process'] = self.process(galaxy, crop_size)
            if not skip_plot:
                result['plot'] = self.plot(galaxy)
            results[galaxy] = result
        return results
