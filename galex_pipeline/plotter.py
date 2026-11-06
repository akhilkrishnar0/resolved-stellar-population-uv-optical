from pathlib import Path
import warnings
import numpy as np
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.wcs import WCS
import matplotlib.pyplot as plt


def plot_fuv_nuv(galaxy_name, base_folder='downloaded_galex', output_folder='galex_plots'):
    """Plot existing stacks using the original class's display stretches."""
    name = galaxy_name.replace(' ', '')
    folder = Path(base_folder) / name / 'cropped'
    outdir = Path(output_folder)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [folder / f'{band}_stacked.fits' for band in ('FUV', 'NUV')]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError(f'Both FUV and NUV stacks are required for plotting {name}')
    fig = plt.figure(figsize=(12, 6))
    try:
        for i, (band, path, stretch, percent) in enumerate(zip(
                ('FUV', 'NUV'), paths, ('sqrt', 'log'), (98.5, 99.5)), 1):
            data, header = fits.getdata(path, header=True)
            if header.get('WGHTTYPE') != 'RRHR':
                warnings.warn(f'{path} is a legacy stack; reprocess to apply rrhr weighting.', stacklevel=2)
            if not np.any(np.isfinite(data)):
                raise ValueError(f'No valid pixels in {path}')
            ax = fig.add_subplot(1, 2, i, projection=WCS(header).celestial)
            ax.imshow(data, origin='lower', cmap='gray',
                      norm=simple_norm(data[np.isfinite(data)], stretch, percent=percent))
            ax.set(title=f'{name} {band}', xlabel='RA', ylabel='Dec')
        fig.tight_layout()
        out = outdir / f'{name}.png'
        fig.savefig(out, dpi=200)
        return out
    finally:
        plt.close(fig)
