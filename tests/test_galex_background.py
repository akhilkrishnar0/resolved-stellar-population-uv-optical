"""Synthetic checks for pre-stack GALEX background handling (no archive data required)."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from galex_pipeline.crop import crop_band
from galex_pipeline.stack import stack_band
from galex_pipeline.background import prepare_background


def _make_pair(root, stem, *, level=2.0, shape=(128, 128), seed=0):
    band = root / 'FUV'
    for kind in ('int', 'intbgsub', 'rrhr'):
        (band / kind).mkdir(parents=True, exist_ok=True)
    w = WCS(naxis=2)
    w.wcs.crpix = [64.0, 64.0]
    w.wcs.cdelt = np.array([-1.0 / 3600, 1.0 / 3600])
    w.wcs.crval = [211.668958, 22.070111]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    header = w.to_header()
    header['BUNIT'] = 'count/s/pixel'
    rng = np.random.default_rng(seed)
    signal = 0.05 + rng.normal(0, 0.003, shape)
    paths = {}
    for kind, data in (('int', signal + level),
                       ('intbgsub', signal),
                       ('rrhr', np.full(shape, 500.0))):
        p = band / kind / f'{stem}-{kind}.fits'
        fits.writeto(p, data, header, overwrite=True)
        paths[kind] = p
    return paths


def test_archive_background_corrected_before_stack(tmp_path):
    first = _make_pair(tmp_path, 'obs1-fd', level=2.0, seed=0)
    second = _make_pair(tmp_path, 'obs2-fd', level=5.0, seed=0)
    pairs = [(x['int'], x['rrhr']) for x in (first, second)]
    crops = crop_band(tmp_path, 'FUV', 211.668958, 22.070111,
                      60 * u.arcsec, pairs=pairs, background_mode='archive')
    output = stack_band(tmp_path/'cropped', 'FUV', pairs=crops)
    stack, hdr = fits.getdata(output['stacked'], header=True)
    reference = fits.getdata(crops[0][0])
    np.testing.assert_allclose(stack, reference, atol=1e-10, equal_nan=True)
    assert hdr['BGSUB']
    assert hdr['BGMODE'] == 'ARCHIVE'


def test_missing_archive_fails_instead_of_stacking_raw(tmp_path):
    paths = _make_pair(tmp_path, 'obs1-fd')
    paths['intbgsub'].unlink()
    with pytest.raises(FileNotFoundError, match='archive'):
        crop_band(tmp_path, 'FUV', 211.668958, 22.070111,
                  60 * u.arcsec, pairs=[(paths['int'], paths['rrhr'])])


def test_local_residual_removal_and_original_preserved(tmp_path):
    paths = _make_pair(tmp_path, 'obs1-fd', shape=(256, 256))
    # Sky-only test field: remove the synthetic 0.05 constant source term
    # before inserting a known 0.04 residual background.
    archive = fits.getdata(paths['intbgsub']) - 0.05
    fits.writeto(paths['intbgsub'], archive + 0.04,
                 fits.getheader(paths['intbgsub']), overwrite=True)
    corrected, model, report = prepare_background(
        paths['int'], paths['rrhr'], mode='local',
        center=SkyCoord(211.668958, 22.070111, unit='deg'),
        galaxy_mask_radius_arcsec=35, mesh_arcmin=0.5)
    assert np.isfinite(model).any()
    assert abs(np.nanmedian(corrected)) < 0.015
    np.testing.assert_allclose(fits.getdata(paths['intbgsub']), archive + 0.04)
    assert report['mode'] == 'local'


def test_profiles_do_not_subtract_again(tmp_path):
    from resolved_uv_optical.profiles import radial_profile
    from astropy.wcs import WCS
    paths = _make_pair(tmp_path, 'obs1-fd')
    data, hdr = fits.getdata(paths['intbgsub'], header=True)
    w = WCS(hdr).celestial
    sky = SkyCoord(211.668958, 22.070111, unit='deg')
    raw, *_ = radial_profile(data, w, sky, 'TEST', 'FUV', 33, 60, subtract_background=False)
    assert np.isfinite(raw['mean_flux_per_pix']).any()
    assert abs(np.nanmedian(raw['mean_flux_per_pix']) - 0.05) < 0.02
