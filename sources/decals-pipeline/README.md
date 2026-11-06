# DECaLS Pipeline

Package prepared from the uploaded `decals_pipeline_class.py`. The unchanged source is retained in `original/`.

## Install
```bash
pip install --upgrade git+https://github.com/akhilkrishnar0/decals-pipeline.git
```
Restart the notebook kernel. Remove or replace an older local `decals_pipeline_class.py` that shadows the installed module.

## Existing-data workflow
```python
from decals_pipeline_class import DecalsPipeline
pipe = DecalsPipeline()
pipe.run(galaxies_new, size=3000, skip_download=True, do_plot=False,
         do_psf_split=False, force_download=False, do_save_split=True)
```
Define `galaxies_new` as your galaxy-name list. This exact call is supported.
It saves g/r/z images and inverse-variance files, without plots or PSF splitting.
Original skip semantics are preserved: downloads are skipped only when the image file exists. Missing images trigger downloading. Existing images with missing inverse variance raise an error during splitting. `force_download=True` always requests downloading.
`size` is forwarded to the external downloader and does not resize existing images. The downloader interprets size in pixels; server limits still apply.

## Required input layout
For a name with spaces removed, e.g. NGC6902:
- `decals_data/downloads/fits/NGC6902.fits`: primary cube `(3, ny, nx)`, g/r/z order.
- `decals_data/downloads/invvar/NGC6902_invvar.fits`: g/r/z inverse variance in HDUs 2/4/6.
- Optional `decals_data/downloads/psf/NGC6902_psf.fits`: three 2-D image HDUs in g/r/z order, required only for plotting or PSF splitting.

These layouts come from the uploaded class; no actual input FITS products were supplied for verification. Ambiguous PSF cubes are rejected rather than assigned a guessed band axis.

## Outputs
- `decals_data/outputs/images/<name>_<band>.fits`
- `decals_data/outputs/invvar/<name>_<band>_invvar.fits`
- Optional `decals_data/plots/<name>.png`
- Optional `decals_data/downloads/psf/<name>_PSF_<band>.fits`

Existing outputs are overwritten. Splitting does not resample or recalibrate pixel values. Celestial WCS is reduced to two dimensions; inverse-variance headers are taken from each band's own HDU. Plotting sigma is computed as `1/sqrt(invvar)` for finite positive weights; invalid weights give NaN, not zero uncertainty. Sigma maps are not exported.

## Included downloader (0.2.0)
`DECaLSQuery` is bundled in `decals_pipeline/query.py`; `from decals_query import DECaLSQuery` remains supported. The uploaded original is retained in `original/decals_query.py`. No separate local downloader is needed.

The original endpoints and output names are preserved. Requests explicitly select `grz`. The image/JPEG URL now uses documented `pixscale=0.262` arcsec/pixel instead of the original `pix=0.25` parameter. This changes the request; old files are not silently redownloaded. The package accepts integer sizes 1–3000 pixels, preserving the uploaded module's intended limit; server limits and availability are not verified by this release.

Downloads use HTTP timeouts, raise HTTP errors, write to temporary files, and inspect FITS structure before replacing the destination. These checks run when you download; no downloads were performed while preparing this release.

### Image and inverse-variance alignment
The uploaded downloader requests a resampled image cube and a `subimage` response for inverse variance. Native subimages can differ in grid and may contain multiple overlapping bricks. Selecting HDUs 2/4/6 is only valid for the legacy seven-HDU single-brick layout. Before splitting or plotting, the class checks that layout, available band labels, array shapes and sampled WCS alignment. It stops on a mismatch rather than silently treating the products as aligned. This integration does not mosaic native bricks or propagate variance through interpolation. A mismatch requires matched native products or a scientifically documented resampling procedure.

See [Legacy Surveys cutout documentation](https://www.legacysurvey.org/svtips/).

## Source corrections
- Bundled downloader and compatibility import; no separate local module required.
- Numeric and legacy sexagesimal SIMBAD coordinate columns supported.
- FITS handles closed after loading.
- Correct band-specific inverse-variance headers and two-dimensional image WCS.
- Consistent three-HDU PSF handling; unsupported layouts raise clear errors.
- Invalid inverse variance does not imply zero uncertainty.
- Stage failures propagate to the notebook.

Source reviewed only. No tests, downloads, or pipeline executions performed. Actual image units, band ordering, astrometric alignment and scientific outputs still need verification with the input FITS products. No license has been selected.
