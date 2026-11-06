# Multiband Cutout Pipeline
Create GALEX FUV/NUV and DECaLS grz FITS cutouts from supplied ICRS positions. A **separate, optional** PSF-matching module is now available. Existing cutout API and raw output names are unchanged.

## Install or update
```python
%pip install --upgrade --force-reinstall "git+https://github.com/akhilkrishnar0/multiband-cutout-pipeline.git"
```
Restart Jupyter's kernel after installation. Local modules of the same name can shadow the installed version.

## 1. Cutouts with a manual galaxy-centre table
Works with an Astropy Table named `centers` containing `Galaxy,RA,DEC` or a pandas DataFrame containing `galaxy,ra_deg,dec_deg`. It matches `UGC9024` to `UGC 9024` without making a SIMBAD request.
```python
from galaxy_catalog import prepare_cutout_table
from multiband_cutout_pipeline import MultiBandCutoutPipeline
df_run=prepare_cutout_table(galaxies_new,distances_new,sizes_new,centers)
pipe=MultiBandCutoutPipeline()
pipe.run(galaxies=df_run["galaxy"].tolist(),ras=df_run["ra_deg"].tolist(),decs=df_run["dec_deg"].tolist(),distances=df_run["distance"].tolist(),sizes=df_run["size"].tolist(),do_plot=True)
```
Supply the lists and centres before this cell. RA/Dec are decimal degrees, distances Mpc, sizes square sides in arcseconds. The adapter checks missing/duplicate centres and impossible coordinate values.

## 2. Standalone PSF matching (optional)
Run **after** producing the cutouts. This is a separate import/module within the same pip-installable distribution, not part of `MultiBandCutoutPipeline.run()`.
```python
from psf_matching_pipeline import PSFMatchingPipeline
psf=PSFMatchingPipeline(base_dir="cutouts",target_fwhm=5.3)
psf.run(galaxies_new,make_plot=True)
```
For a single object:
```python
from psf_matching_pipeline import process_galaxy
process_galaxy("UGC9024",base_dir="cutouts",make_plot=True)
```

Input: `cutouts/<galaxy>/g.fits,r.fits,z.fits,FUV.fits,NUV.fits` (existing bands only). Output: `<band>_psfmatch.fits` and `<galaxy>_psfmatched.png`; raw input files remain unchanged. Existing PSF-matched products with these names are overwritten.

The Gaussian approximations default to assumed native FWHM (arcsec) `g=1.3, r=1.2, z=1.1, FUV=4.2, NUV=5.3`, target `5.3`. Kernel sigma is `sqrt(FWHM_target**2-FWHM_native**2)/(2.35482*pixel_scale)`; WCS-derived pixel scale is checked. NaN-aware normalized convolution mitigates blank-pixel dilution; pixels near missing-data boundaries are not guaranteed to conserve integrated flux. A wider native PSF than target raises an error rather than being mislabeled as matched. Change `native_fwhm` if the actual measured PSFs differ:
```python
psf=PSFMatchingPipeline(native_fwhm={"g":1.3,"r":1.2,"z":1.1,"FUV":4.2,"NUV":5.3})
```
**Science caveat:** these are illustrative circular-Gaussian FWHM assumptions, not a measured PSF-homogenization solution. Real GALEX/DECaLS PSFs have non-Gaussian wings and may vary across the field; quantitative resolved colours/photometry require an empirical-kernel and aperture-level validation. This stage does **not** align images onto a common WCS grid: PSF matching is not reprojection.

## Inputs and original cutout outputs
- `downloaded_galex/<name>/cropped/FUV_stacked.fits` and `NUV_stacked.fits`
- `decals_data/outputs/images/<name>_g.fits` (and `r`, `z`)
- Output directory `cutouts/<name>/`: FUV, NUV, g, r, z FITS and optional plots.

Names lose spaces in filesystem paths. Use `galex_dir`, `decals_dir`, `out_dir` in the cutout constructor to override paths. For PSF matching use the same output directory as `base_dir`.

## Existing cutout-pipeline limitations
The original cutout implementation is unchanged in this update. Its RGB arrays are not WCS reprojected and may differ in shape or registration, its scale bars use fixed assumed pixel scales, its original resolver expects an older SIMBAD response, and some failures are logged rather than raised. The new `prepare_cutout_table` avoids SIMBAD and fixes the documented Astropy/pandas-name mismatch for manual centre inputs.

Source reviewed, not scientifically validated on real image stacks; no production imaging runs or quantitative flux/PSF verification have been performed. No repository license has been selected.
