# Resolved Stellar Population UV–Optical

Combined codebase for resolved-scale galaxy studies using GALEX FUV/NUV and DECaLS g/r/z imaging. One installation includes the existing download/preparation packages, multiband cutouts, galaxy-centre table helper, and Gaussian PSF matching.

## Current scope

This is the shared home for the study's complete code as it is assembled. The current code covers image preparation, approximate PSF matching, point-source masking and scalar residual UV-background subtraction. It does **not yet include** spatial background modelling, extinction corrections, spatial binning, resolved photometry, stellar population fitting, stellar-mass/SFR maps, or scientific uncertainty propagation. Those original analysis codes must be added before calling this a complete population-analysis pipeline.

## Install

```python
%pip install git+https://github.com/akhilkrishnar0/resolved-stellar-population-uv-optical.git
```

Restart the kernel. This distribution replaces the three separate installations: uninstall galex_pipeline, glsb-decals-pipeline and glsb-multiband-cutout-pipeline first if installed, because their import modules overlap. Remove older same-named local Python files that shadow the package.

## Recommended sequence: review both surveys, then process

Run these cells separately, inspecting the files between steps:

```python
from resolved_uv_optical import ResolvedPopulationPipeline
pipe = ResolvedPopulationPipeline()
pipe.review_galex(catalog)  # download and review only
```

Inspect `galex_review/<galaxy>/inputs.csv` and FUV/NUV PNGs; enter rejected
intensity filenames in `galex_review/exclusions.json` or pass them at processing.

```python
pipe.review_decals(catalog, size=3000)  # download and review only
```

Inspect `decals_review/<galaxy>/grz.png`: three science bands in one row with the
catalogue centre marked. `files.csv` lists the downloaded science, inverse-variance
and PSF products and their presence. The preview reads the science cube directly;
it does not split images or require aligned inverse-variance grids. This is a visual
science-image review, not validation of variance or PSF calibration.
Use `download=False` in either review method to inspect existing local downloads.

```python
exclude = {"UGC9024": []}  # exact rejected GALEX intensity filenames
report = pipe.process(catalog, galex_exclude=exclude)
```

Processing uses existing downloads: GALEX selection/cropping/stacking, DECaLS
splitting, five-band cutouts, plots and Gaussian PSF matching. Saved GALEX exclusions
are used when `galex_exclude` is omitted. Review and processing are separate calls;
manual approval is your decision to run the next cell, not an automatic assessment.
For an unacceptable DECaLS cutout, replace/redownload it or remove that galaxy from
the processing catalogue. There is one downloaded g/r/z cube per galaxy, not a list
of individual DECaLS exposures to reject.

PSF matching does not yet put all bands on a common WCS/pixel grid. DECaLS inverse-variance splitting is optional and retains strict grid checks. A 3000-pixel
DECaLS cutout at 0.262 arcsec/pixel spans about 786 arcsec; larger requested final
cutouts will have uncovered optical edges.

## Use existing prepared images

```python
from resolved_uv_optical import ResolvedPopulationPipeline, prepare_cutout_table
catalog = prepare_cutout_table(galaxies_new, distances_new, sizes_new, centers)
pipe = ResolvedPopulationPipeline(base_dir=".")
report = pipe.run(catalog, make_cutouts=True, do_plot=False)
```

The catalogue helper accepts a pandas DataFrame or Astropy Table for centres, normalizes names for matching, and rejects missing coordinates. It needs galaxy/ra_deg/dec_deg (or Galaxy/RA/DEC) columns. Distances are Mpc, sizes are arcseconds, RA/Dec are ICRS degrees. The workflow removes spaces from output galaxy names.

## Enable earlier preparation stages

```python
report = pipe.run(catalog, download_galex=True, process_galex=True,
    download_decals=True, split_decals=True, make_cutouts=True, do_plot=False,
    decals_size_pixels=3000)
```

Every combined-workflow stage flag is literal: False never runs the stage. New DECaLS downloads overwrite named products; enabled processing overwrites outputs. Keep copies of previous scientific outputs for comparison. The underlying legacy classes remain directly importable and retain their original flags and behaviour.

GALEX uses per-pixel rrhr weights. The intermediate crop defaults to each final angular size; set galex_crop_arcsec explicitly for a larger field. Supplied RA/Dec are used for GALEX processing, DECaLS downloads and final cropping. The imported GALEX downloader still resolves the galaxy name and searches within its existing 0.1-degree radius; archive coverage should be reviewed.

## Review GALEX observations before stacking

First download and produce the review table and full-field PNGs. `review_only=True`
ends each galaxy after review, regardless of other stage flags; no stacks, DECaLS
processing, cutouts or PSF matching are run.

```python
pipe.run(catalog, download_galex=True, review_only=True)
```

For already downloaded files use `pipe.run(catalog, review_only=True)`.
Open `galex_review/<galaxy>/inputs.csv` and the PNGs in its FUV/NUV folders.
Each PNG shows the original intensity image and matching rrhr response map.
A red cross and cyan circle mark the supplied catalogue centre; an explicit label
identifies centres outside the field. The circle is a visual locator, not a photometric
aperture. Percentile/asinh stretching and display subsampling affect previews only.
Missing response maps appear in the table and previews. Preview failures are recorded
in the table and raised, so an incomplete review is not silently accepted.

Give exact **intensity FITS filenames** from the table, separately for each galaxy:

```python
exclude = {"UGC9024": ["actual-observation-fd-int.fits"]}  # replace with real filename
report = pipe.run(catalog, process_galex=True, galex_exclude=exclude,
    split_decals=True, make_cutouts=True, do_plot=True, match_psf=True)
```

Use `exclude = {}` if nothing is rejected. Alternatively edit the automatically
created `galex_review/exclusions.json`, for example a JSON object mapping each galaxy
to a list of filenames. It is loaded on every GALEX review/processing run. An explicit
`galex_exclude` entry overrides the saved list for that galaxy for this run only;
`{"UGC9024": []}` explicitly clears that galaxy's saved exclusions for the run.
Unknown filenames raise an error. Rejections are band-specific and do not delete files.
Rejected intensity/response pairs never enter cropping or stacking. Empty selected
bands or missing responses for selected images stop processing.

The table starts with candidate/excluded rows and records stacked/no_overlap after
successful stacking. Timestamped input-table snapshots and effective exclusions are
also saved in run_reports. Existing stale crops are never included in the new stack.
Review-only runs do not change existing stacks; after changing exclusions rerun
`process_galex=True` and regenerate downstream cutouts/PSF products. Set
`galex_previews=False` to skip PNG regeneration once visual review is complete.
Only stack-input intensity/rrhr pairs are previewed; sky-background and background-
subtracted archive products are not inputs to this stack and are not displayed.

## DECaLS science splitting and uncertainty maps

The combined workflow now splits science images independently of inverse variance:

```python
pipe.run(catalog, split_decals=True, make_cutouts=False)
```

This fixes science-image splitting being blocked by an unexpected inverse-variance
HDU layout. Both `run` and `process` default to `split_decals_invvar=False`. Existing
inverse-variance output files are unchanged, not validated or refreshed, and the run
report records `decals_inverse_variance: not_processed`. Science cutouts and Gaussian
PSF matching do not consume those inverse-variance files.

To request uncertainty-map splitting explicitly, pass `split_decals_invvar=True`.
This retains the existing seven-HDU, shape and WCS checks and will still reject
unsupported files. Supporting other layouts or multiple bricks requires inspecting
their actual HDU headers and implementing the appropriate selection/mosaicking and
variance propagation; disabling this check would not make the maps valid.
The legacy `DecalsPipeline.save_split_data` retains its strict default; pass
`include_invvar=False` there for science-only splitting.

## Optional PSF matching

```python
report = pipe.run(catalog, make_cutouts=False, match_psf=True, do_plot=False,
    target_fwhm=5.3)
```

The imported module assumes Gaussian PSFs with FWHM FUV=4.2, NUV=5.3, g=1.3, r=1.2, z=1.1 arcsec. These are assumptions, not measurements for each observation. Supply native_fwhm as a band-to-FWHM dictionary for your data. Matching is opt-in. It does not align pixel grids or propagate variance; normalized convolution can fill masked holes. PSF outputs are named <band>_psfmatch.fits.

## Mask sources and subtract residual UV background

After PSF matching, use the supplied masking script through the combined class:

```python
mask_report = pipe.mask_sources(approved, protect_galaxy=False,
    allow_dao_fallback=False, do_bgsub=True, make_plots=True)
```

This example deliberately disables the protected ellipse because measured galaxy
geometries have not been supplied. DAO detections in the g-band reference are matched
to Gaia DR3 objects passing the script's proper-motion/parallax SNR cuts. Inspect all
masks, particularly inside the galaxy. This is not a complete background-galaxy mask;
add manual apertures for unrelated objects that the star-selection step misses.

For a protected region, provide measured `galaxy_sma_arcsec`, `galaxy_smb_arcsec` and
`galaxy_pa_deg` via `per_galaxy={"NAME": {...}}`, or common keyword settings if appropriate.
No galaxy dimensions are inferred from the cutout size. PA retains the uploaded
script convention: **east toward north**, not the usual north-through-east catalogue
PA; convert using `90 - PA_north_through_east` modulo 180 degrees. Semiaxes are arcseconds.
Automatic masks avoid the protected region; manual masks override that protection.

`manual_circles_refpix=[(x,y,r)]` and `manual_ellipses_refpix=[(x,y,a,b,theta_deg)]`
are specified separately per galaxy in zero-based reference-image pixels; ellipse
angles are from pixel +x toward +y. `per_galaxy` overrides common settings.

Inputs default to `cutouts/<galaxy>/<band>_psfmatch.fits`. Outputs are separate under
`masked/<galaxy>/`: masked science FITS, binary masks, source catalogue, diagnostic PNGs,
background-subtracted FITS, and `masking_report.json` with settings and background statistics.
Original inputs are preserved. The script maps the reference mask through WCS to each
band's own grid; it does not reproject the science images onto a common grid.

Background subtraction follows the supplied method: sigma-clipped scalar median of
finite, unmasked pixels beyond 0.70 of the image-centred corner radius, FUV/NUV only.
This region is weighted toward image corners, not a measured galaxy-free sky annulus.
Check that it lies outside galaxy emission and inside valid coverage before using the
result. Optical `*_bgsub.fits` files are unchanged masked copies, with `BGSUB=False`.
Use `do_bgsub=False` for masking alone or change the supplied background settings.
Insufficient background pixels raise an error in the wrapper; any written copy has
`BGSUB=False`. No uncertainty propagation is added.

The original script is archived in `sources/masking/`. Integration fixes include
missing-input checks, finite WCS handling, RA-wrap-safe offsets, masked-array clipping,
and an undefined FITS background keyword instead of an invalid NaN header value.
DAO-only fallback is now opt-in; without it a Gaia query failure stops processing.
Gaia positions are not propagated to the image epoch. This post-PSF mask does not undo
contaminant flux already spread by convolution, so inspect mask radii/stellar wings.

## Layout and provenance

- galex_pipeline/: GALEX downloads, paired crops and response-weighted stacks.
- decals_pipeline/: DECaLS downloads and splitting with grid checks.
- multiband_cutout_pipeline.py: original five-band crops and display plots.
- psf_matching_pipeline.py: imported Gaussian PSF matching.
- galaxy_catalog.py: input catalogue helper.
- resolved_uv_optical/: combined orchestration.
- examples/: notebook using one catalogue and one workflow.
- sources/: original READMEs, package metadata and archived source files.
- SOURCE_VERSIONS.json: exact upstream snapshots.

Imported modules originate from the recorded source snapshots. GALEX cropping now also accepts an explicit selected input-pair list. The combined wrapper adds catalogue validation, explicit stage controls, strict band checks and run reporting. It calls the existing crop method directly so a failed crop cannot be swallowed by the old per-galaxy loop.

## Outputs

Existing paths are retained relative to base_dir: downloaded_galex/, decals_data/, cutouts/, galex_plots/. Each combined run writes its input table and a JSON status/options report under run_reports/. A failure is recorded and raised; earlier completed stages may already have written outputs.

## Scientific limits to resolve

DECaLS resampled science cutouts and native subimage inverse variances may have different grids; the existing code stops on mismatch. No automatic variance resampling or brick mosaicking has been introduced. Multiband RGB plotting still assumes aligned arrays, so plotting is off by default. Science cutouts retain native sampling, without a common grid. Response/coverage/inverse-variance cutouts are not yet carried through the multiband stage. Original hard-coded display scale bars and header-handling limitations remain documented in sources/. No physical-population outputs should be inferred from these preparation products alone.

Source integration reviewed only. No tests, pipeline execution, archive downloads or scientific-output validation performed. No licence has been selected.


