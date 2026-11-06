# GALEX Pipeline

Version 0.2.0 — updated 5 November 2026.

Download GALEX FUV/NUV products, crop matched intensity/relative-response images,
and combine images using the effective response at each pixel. Includes the
`GalexPipeline` class used in GLSB notebooks.

## Installation

```bash
git clone https://github.com/akhilkrishnar0/galex-pipeline.git
cd galex-pipeline
pip install -e .
```

For an existing clone, run `git pull --ff-only`, then `pip install -e .`.
Restart your notebook kernel after upgrading. Python 3.9 or newer is required.
Jupyter itself is installed separately if needed.
If an older `galex_pipeline_class.py` is beside your notebook, replace it with
this repository's compatibility module or remove it so it cannot shadow the
installed module.

## Rebuild existing stacks with spatial response weights

```python
from galex_pipeline_class import GalexPipeline
from astropy import units as u

pipe = GalexPipeline()
# Define your own complete sample here; this is only an example.
galaxies_new = ['NGC6902']
pipe.run(galaxies_new, crop_size=800*u.arcsec,
         skip_download=False, skip_process=False, skip_plot=False)
```

Choose the crop size used for your analysis. The class retains its original
`crop_size=300` default, meaning **300 by 300 pixels**. An angular Quantity such
as `800*u.arcsec` means **800 by 800 arcsec**. The functional API
`process_galaxy` retains its 800-arcsec default. A rerun overwrites the named
stack/map outputs and plots; preserve old results separately before comparing.

All skip flags are now unconditional: **True skips, False runs**. To plot only:

```python
pipe.run(galaxies_new, skip_download=True, skip_process=True, skip_plot=False)
```

This does not upgrade old stacks. Legacy stacks generate a warning when plotted.
`is_done()` checks the presence of new-format products, not whether the crop size
or input selection has changed. Processing is explicit and never automatically
skipped based on old files. Custom `base_dir` is passed through every stage.
Failures propagate rather than being printed as successful completion.

## Scientific weighting

GALEX `int` is already a count-rate image: `int = cnt / rrhr`.
`rrhr` is the high-resolution relative response (seconds), including the spatial
exposure and flat-field response. The lower-resolution `exp` product does not
include the flat field and is not substituted for `rrhr`.

Each intensity image must have its exact observation/band-matched `rrhr` file.
The code checks shape and sampled WCS alignment, then applies identical crop
slices to both. Crops have a fixed requested size with NaN intensity and zero
response outside the native image. Images without overlap are omitted.

For valid input pixels, the algorithm computes:

```text
numerator = sum(reproject(int * rrhr))
weight    = sum(reproject(rrhr))
stack     = numerator / weight
```

Both arrays use the same bilinear reprojection onto the first current cutout's
WCS and shape. Inputs must have matching pixel areas (standard GALEX images do);
differing areas raise an error instead of silently mixing counts/s/pixel scales.
Nonfinite intensities/responses and nonpositive responses receive zero weight.
Uncovered output pixels are NaN, not zero. Header `EXPTIME` never sets a weight.
`TOTEXPT`, when available for all inputs, is nominal observation metadata only.

The response output is a summed **effective response**, not a pure shutter-time
map or an inverse-variance map. No extra exposure division is applied to `int`.
Reprojection interpolates images and introduces correlated noise; the products
are not a propagated uncertainty model or a demonstration of exact photon-count
conservation after resampling.

## Downloads and input selection

The downloader searches within 0.1 degree by default, filters GALEX observations,
and retrieves `int`, `rrhr`, `skybg`, and `intbgsub` products. Each download call
queries MAST again and retrieves missing exact products even when other FITS files
already exist. Downloads use temporary files and retry failures. Existing
nonempty files are reused; this is not a checksum or corruption audit.
A JSON manifest records the selected archive URIs after pairing succeeds.
Missing `rrhr` files cause an error; there is no scalar-exposure fallback.

The search radius is configurable through `pipe.download_galex(galaxy, size=...)`.
A cone search is not a proof that all observations overlapping a large crop have
been found. Review the product manifest and ensure the selected observations are
independent: do not combine an archive coadd with visits contributing to that
coadd. The downloader deduplicates identical URIs, not overlapping photon content.
Only `int`/`rrhr` pairs enter the stack; background products are retained for later
analysis. Additional artifact masks and any low-response threshold must be chosen
and applied as part of the scientific analysis; no arbitrary threshold is imposed.

## Outputs

Under `downloaded_galex/<galaxy>/`:

- `FUV/` and `NUV/`: `int/`, `rrhr/`, `skybg/`, `intbgsub/`.
- `cropped/FUV/` and `cropped/NUV/`: paired cropped inputs.
- `cropped/<band>_stacked.fits`: intensity, counts/s/pixel.
- `cropped/<band>_rrhr.fits`: summed effective response, seconds; zero without coverage.
- `cropped/<band>_coverage.fits`: number of images with positive valid reprojected response.
- `download_manifest.json`: selected filenames and archive URIs.

Stacks include `WGHTTYPE=RRHR`, `PIPEVER`, `NCOMBINE`, and input filenames in HISTORY.
The current crop-pair list is passed directly to stacking, excluding stale cutouts.
Plots are saved to `galex_plots/<galaxy>.png`: FUV square-root/98.5% and NUV
logarithmic/99.5% display normalization. Both bands are required for paired plots.
A single available band can be processed with `skip_plot=True`.

## Scope and verification status

This package prepares images. It does not perform background subtraction,
foreground-source masking, extinction correction, photometry, SFR estimation,
or uncertainty propagation. Background treatment and spatially varying noise
remain necessary for GLSB measurements, particularly near field edges.

The 0.2.0 changes were reviewed at source level. No tests, archive downloads,
pipeline executions, or scientific-output validation were performed for this
update. Compare rebuilt aperture fluxes and radial profiles with the previous
stacks before adopting revised scientific measurements.

## References

- [GALEX imaging products](https://www.galex.caltech.edu/researcher/gr5_docs/techdoc-ch4.html)
- [MAST GALEX data FAQ](https://galex.stsci.edu/gr6/?page=ddfaq)

The previous README mentioned MIT licensing, but this repository has no standalone
license grant. An explicit license should be chosen by the author before reuse terms
are assumed.
