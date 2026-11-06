from pathlib import Path
from astropy.coordinates import SkyCoord
from astropy import units as u
from .io_utils import unzip_galex_files, organize_fits, image_pairs
from .crop import crop_band
from .stack import stack_band


def process_galaxy(gname, root='downloaded_galex', crop_size=800*u.arcsec):
    folder = Path(root) / gname.replace(' ', '')
    if not folder.is_dir():
        raise FileNotFoundError(f'Download data first: {folder}')
    pos = SkyCoord.from_name(gname)
    unzip_galex_files(folder)
    organize_fits(folder)
    available = {band: image_pairs(folder, band) for band in ('FUV', 'NUV')}
    if not any(available.values()):
        raise FileNotFoundError(f'No matched intensity/rrhr data in {folder}')
    result = {}
    # Pass the current pair list explicitly so old cutouts cannot enter a rerun.
    for band in ('FUV', 'NUV'):
        pairs = crop_band(folder, band, pos.ra.deg, pos.dec.deg, crop_size)
        if not pairs and (folder / 'cropped' / f'{band}_stacked.fits').exists():
            raise ValueError(f'No current {band} coverage; an older stack exists. Move old outputs before continuing.')
        result[band] = stack_band(folder / 'cropped', band, pairs=pairs)
    return result


def process_many(galaxy_list, root='downloaded_galex', crop_size=800*u.arcsec):
    return {g: process_galaxy(g, root, crop_size) for g in galaxy_list}
