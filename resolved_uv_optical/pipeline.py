"""One workflow around the existing GLSB preparation modules."""
from pathlib import Path
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from galex_pipeline import GalexPipeline
from decals_pipeline import DecalsPipeline
from multiband_cutout_pipeline import MultiBandCutoutPipeline
from psf_matching_pipeline import PSFMatchingPipeline

class ResolvedPopulationPipeline:
    def __init__(self, base_dir=".", layer="ls-dr9"):
        self.base_dir = Path(base_dir).resolve()
        self.galex = GalexPipeline(base_dir=self.base_dir/"downloaded_galex", plot_dir=self.base_dir/"galex_plots")
        self.decals = DecalsPipeline(base_dir=self.base_dir/"decals_data", layer=layer)
        self.cutouts = MultiBandCutoutPipeline(galex_dir=self.galex.base_dir,
            decals_dir=self.decals.base_dir/"outputs"/"images", out_dir=self.base_dir/"cutouts")

    def mask_sources(self, catalog, *, per_galaxy=None, **settings):
        """Mask PSF-matched bands and optionally subtract the residual UV background.

        Geometry uses the uploaded script convention: PA from east toward north.
        Manual coordinates are zero-based reference-image pixels.
        """
        from dataclasses import asdict
        from .masking import MaskConfig, run_masking, get_path
        table = catalog.to_pandas() if hasattr(catalog, "to_pandas") else pd.DataFrame(catalog).copy()
        required = {"galaxy", "ra_deg", "dec_deg"}
        if table.empty or not required.issubset(table.columns):
            raise ValueError("Masking needs galaxy, ra_deg and dec_deg columns")
        names = table.galaxy.astype(str).str.replace(" ", "", regex=False)
        if table.galaxy.isna().any() or names.duplicated().any() or any(n in ("", ".", "..") or "/" in n or "\\" in n for n in names):
            raise ValueError("Invalid or duplicate galaxy names")
        per_galaxy = {} if per_galaxy is None else per_galaxy
        unknown = set(per_galaxy)-set(names)
        if unknown:
            raise ValueError(f"Unknown per_galaxy names: {sorted(unknown)}")
        configs = []
        for name, row in zip(names, table.itertuples(index=False)):
            ra, dec = float(row.ra_deg), float(row.dec_deg)
            if not np.isfinite([ra, dec]).all() or not 0 <= ra < 360 or not -90 <= dec <= 90:
                raise ValueError(f"Invalid centre for {name}")
            options = dict(settings)
            options.update(per_galaxy.get(name, {}))
            reserved = {"base_dir", "out_dir", "galaxy", "center_ra", "center_dec"} & set(options)
            if reserved:
                raise ValueError(f"Use the catalogue/base_dir for {sorted(reserved)}")
            if options.get("protect_galaxy", True):
                for key in ("galaxy_sma_arcsec", "galaxy_smb_arcsec", "galaxy_pa_deg"):
                    if key not in options or options[key] is None or not np.isfinite(options[key]):
                        raise ValueError(f"Supply {key} for {name}, or explicitly set protect_galaxy=False")
                if min(options["galaxy_sma_arcsec"], options["galaxy_smb_arcsec"]) <= 0:
                    raise ValueError("Protected semiaxes must be positive")
            options.setdefault("file_pattern", "{band}_psfmatch.fits")
            config = MaskConfig(base_dir=self.cutouts.out_dir/name,
                out_dir=self.base_dir/"masked"/name, galaxy=name,
                center_ra=ra, center_dec=dec, **options)
            if not 0 <= config.bg_inner_frac < 1 or config.bg_statistic not in ("mean", "median"):
                raise ValueError("Invalid background settings")
            missing = [str(get_path(config, b)) for b in set(config.bands) | {config.reference_band} if not get_path(config, b).is_file()]
            if missing:
                raise FileNotFoundError("Missing masking inputs: " + ", ".join(missing))
            configs.append(config)
        reports = {}
        for config in configs:
            path = config.out_dir/"masking_report.json"
            report = {"galaxy": config.galaxy, "status": "running", "settings": asdict(config)}
            def save_report():
                path.write_text(json.dumps(report, indent=2, default=str)+"\n")
            save_report()
            try:
                result = run_masking(config)
                report.update(status="complete", stats=result["stats"], saved=result["saved"],
                    source_table_csv=result["source_table_csv"], reference_mask_fits=result["reference_mask_fits"])
                if any(p["background_status"].startswith("background_failed") for p in result["saved"]):
                    report["status"] = "background_failed"
                    raise ValueError(f"Insufficient background pixels for {config.galaxy}; inspect masking_report.json")
            except Exception as exc:
                report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                save_report()
            reports[config.galaxy] = report
        return reports

    def review_galex(self, catalog, *, download=True, galex_exclude=None):
        """Step 1: download (optional), preview GALEX, then stop."""
        return self.run(catalog, download_galex=download, review_only=True,
            galex_exclude=galex_exclude)

    def review_decals(self, catalog, *, download=True, size=3000):
        """Step 2: download (optional), preview DECaLS g/r/z, then stop."""
        return self.run(catalog, download_decals=download, decals_review_only=True,
            decals_size_pixels=size)

    def process(self, catalog, *, galex_exclude=None, do_plot=True,
                match_psf=True, target_fwhm=5.3, native_fwhm=None, split_decals_invvar=False):
        """Step 3: process locally reviewed data; never download."""
        return self.run(catalog, process_galex=True, split_decals=True,
            make_cutouts=True, do_plot=do_plot, match_psf=match_psf,
            target_fwhm=target_fwhm, native_fwhm=native_fwhm,
            galex_exclude=galex_exclude, galex_previews=False,
            split_decals_invvar=split_decals_invvar)

    def run(self, catalog, *, download_galex=False, process_galex=False,
            download_decals=False, split_decals=False, make_cutouts=True,
            do_plot=False, match_psf=False, target_fwhm=5.3, native_fwhm=None,
            galex_crop_arcsec=None, decals_size_pixels=3000,
            review_only=False, galex_previews=True, galex_exclude=None,
            decals_review_only=False, split_decals_invvar=False):
        """catalog columns: galaxy, ra_deg, dec_deg, distance (Mpc), size (arcsec).
        Every stage flag is literal: False never invokes that stage.
        Existing named products are overwritten by enabled stages.
        """
        if decals_review_only and (review_only or download_galex or process_galex):
            raise ValueError("Run GALEX and DECaLS review separately")
        table = catalog.to_pandas() if hasattr(catalog, "to_pandas") else pd.DataFrame(catalog).copy()
        columns = ["galaxy", "ra_deg", "dec_deg", "distance", "size"]
        if any(c not in table for c in columns) or table.empty:
            raise ValueError(f"Supply a nonempty catalog with columns {columns}")
        for c in columns[1:]:
            table[c] = pd.to_numeric(table[c], errors="raise")
        values = table[columns[1:]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not table.ra_deg.between(0,360,inclusive="left").all() or not table.dec_deg.between(-90,90).all() or (table.distance<=0).any() or (table["size"]<=0).any():
            raise ValueError("Invalid coordinates, distances or angular sizes")
        if table.galaxy.isna().any():
            raise ValueError("Galaxy names cannot be missing")
        names = table.galaxy.astype(str).str.replace(" ", "", regex=False)
        if names.duplicated().any() or names.eq("").any() or names.isin([".", ".."]).any() or names.str.contains(r"[/\\\\]", regex=True).any():
            raise ValueError("Galaxy names must be unique nonempty file-safe names")
        if galex_crop_arcsec is not None and (not np.isfinite(galex_crop_arcsec) or galex_crop_arcsec < table["size"].max()):
            raise ValueError("galex_crop_arcsec must cover every requested cutout")
        table["galaxy"] = names
        from .galex_review import load_exclusions, prepare_review
        exclusions = load_exclusions(self.base_dir, galex_exclude) if (download_galex or process_galex or review_only) else {}
        reports = self.base_dir/"run_reports"
        reports.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        table.to_csv(reports/f"{stamp}_inputs.csv", index=False)
        report = {"started_utc": stamp, "status": "running", "galaxies": {},
            "options": dict(download_galex=download_galex, process_galex=process_galex,
                download_decals=download_decals, split_decals=split_decals,
                make_cutouts=make_cutouts, do_plot=do_plot, match_psf=match_psf,
                target_fwhm=target_fwhm, native_fwhm=native_fwhm,
                galex_crop_arcsec=galex_crop_arcsec, decals_size_pixels=decals_size_pixels,
                review_only=review_only, galex_previews=galex_previews, galex_exclude=exclusions,
                decals_review_only=decals_review_only, split_decals_invvar=split_decals_invvar)}
        report_path = reports/f"{stamp}.json"
        def save():
            report_path.write_text(json.dumps(report, indent=2, default=str)+"\n")
        save()
        try:
            for row in table.itertuples(index=False):
                name = row.galaxy
                stages = report["galaxies"][name] = {}
                if download_galex:
                    self.galex.run_download(name)
                    stages["galex_download"] = "complete"
                if download_galex or process_galex or review_only:
                    folder = self.galex.base_dir/name
                    review_dir = self.base_dir/"galex_review"/name
                    inventory = prepare_review(folder, review_dir, name, row.ra_deg, row.dec_deg,
                        exclusions.get(name, []), previews=galex_previews)
                    stages["galex_review"] = str(review_dir/"inputs.csv")
                    if inventory.empty:
                        raise FileNotFoundError(f"No GALEX intensity images for {name}")
                    errors = inventory.loc[inventory.preview_error.ne(""), "filename"].tolist()
                    if errors:
                        raise ValueError(f"GALEX preview errors; see inputs.csv: {errors}")
                if review_only:
                    save()
                    continue
                if process_galex:
                    from galex_pipeline.crop import crop_band
                    from galex_pipeline.stack import stack_band
                    selected = {}
                    for band in ("FUV", "NUV"):
                        selected[band] = inventory.loc[(inventory.band == band) & ~inventory.excluded]
                        if selected[band].empty:
                            raise ValueError(f"No selected {band} images for {name}")
                        missing = selected[band].loc[~selected[band].response_exists, "filename"].tolist()
                        if missing:
                            raise FileNotFoundError(f"Missing GALEX response maps: {missing}")
                    for band, inputs in selected.items():
                        raw_pairs = [(Path(r.intensity_path), Path(r.response_path)) for r in inputs.itertuples()]
                        pairs = crop_band(folder, band, row.ra_deg, row.dec_deg,
                            float(galex_crop_arcsec or row.size)*u.arcsec, pairs=raw_pairs)
                        if not pairs:
                            raise ValueError(f"No current {band} coverage for {name}")
                        stack_band(folder/"cropped", band, pairs=pairs)
                        used = {p.name.replace("_crop.fits", ".fits") for p, _ in pairs}
                        for i in inputs.index:
                            inventory.at[i, "status"] = "stacked" if inventory.at[i, "filename"] in used else "no_overlap"
                        inventory.to_csv(review_dir/"inputs.csv", index=False)
                        inventory.to_csv(reports/f"{stamp}_{name}_galex_inputs.csv", index=False)
                    stages["galex_process"] = "complete"
                if download_decals:
                    from decals_pipeline.query import DECaLSQuery
                    DECaLSQuery(output_dir=str(self.decals.download_dir), layer=self.decals.layer).query_region(
                        SkyCoord(row.ra_deg, row.dec_deg, unit="deg"), size=decals_size_pixels,
                        download_type="all", galid=name)
                    stages["decals_download"] = "complete"
                if decals_review_only:
                    from .decals_review import prepare_decals_review
                    stages["decals_review"] = prepare_decals_review(self.decals.download_dir,
                        self.base_dir/"decals_review"/name, name, row.ra_deg, row.dec_deg)
                    save()
                    continue
                if split_decals:
                    self.decals.save_split_data(name, include_invvar=split_decals_invvar)
                    stages["decals_split"] = "complete"
                    stages["decals_inverse_variance"] = "split_and_grid_checked" if split_decals_invvar else "not_processed"
                if make_cutouts:
                    files = self.cutouts.get_files(name)
                    missing = [str(p) for p in files.values() if not p.is_file()]
                    if missing:
                        raise FileNotFoundError("Missing science images: "+", ".join(missing))
                    center = SkyCoord(row.ra_deg, row.dec_deg, unit="deg")
                    # Complete all crops before writing; do not silently accept a failed band.
                    crops = {band:self.cutouts.crop(path, center, row.size) for band,path in files.items()}
                    out = self.cutouts.out_dir/name
                    out.mkdir(parents=True, exist_ok=True)
                    for band,(data,header) in crops.items():
                        fits.writeto(out/f"{band}.fits", data, header, overwrite=True)
                    stages["cutouts"] = "complete"
                if do_plot or match_psf:
                    missing = [str(self.cutouts.out_dir/name/f"{b}.fits") for b in ("FUV","NUV","g","r","z") if not (self.cutouts.out_dir/name/f"{b}.fits").is_file()]
                    if missing:
                        raise FileNotFoundError("Missing cutouts: "+", ".join(missing))
                if do_plot:
                    self.cutouts.plot(name, distance_mpc=row.distance)
                    stages["plots"] = "complete"
                if match_psf:
                    PSFMatchingPipeline(base_dir=self.cutouts.out_dir, target_fwhm=target_fwhm,
                        native_fwhm=native_fwhm).process_galaxy(name, make_plot=do_plot)
                    stages["psf_matching"] = "complete"
                save()
            report["status"] = "complete"
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            save()
        return report


