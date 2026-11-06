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

    def run(self, catalog, *, download_galex=False, process_galex=False,
            download_decals=False, split_decals=False, make_cutouts=True,
            do_plot=False, match_psf=False, target_fwhm=5.3, native_fwhm=None,
            galex_crop_arcsec=None, decals_size_pixels=3000):
        """catalog columns: galaxy, ra_deg, dec_deg, distance (Mpc), size (arcsec).
        Every stage flag is literal: False never invokes that stage.
        Existing named products are overwritten by enabled stages.
        """
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
        reports = self.base_dir/"run_reports"
        reports.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        table.to_csv(reports/f"{stamp}_inputs.csv", index=False)
        report = {"started_utc": stamp, "status": "running", "galaxies": {},
            "options": dict(download_galex=download_galex, process_galex=process_galex,
                download_decals=download_decals, split_decals=split_decals,
                make_cutouts=make_cutouts, do_plot=do_plot, match_psf=match_psf,
                target_fwhm=target_fwhm, native_fwhm=native_fwhm,
                galex_crop_arcsec=galex_crop_arcsec, decals_size_pixels=decals_size_pixels)}
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
                if process_galex:
                    # Use supplied coordinates for preprocessing as well as final cutouts.
                    from galex_pipeline.io_utils import unzip_galex_files, organize_fits, image_pairs
                    from galex_pipeline.crop import crop_band
                    from galex_pipeline.stack import stack_band
                    folder = self.galex.base_dir/name
                    unzip_galex_files(folder)
                    organize_fits(folder)
                    for band in ("FUV", "NUV"):
                        if not image_pairs(folder, band):
                            raise FileNotFoundError(f"No {band} input pairs for {name}")
                    for band in ("FUV", "NUV"):
                        pairs = crop_band(folder, band, row.ra_deg, row.dec_deg,
                            float(galex_crop_arcsec or row.size)*u.arcsec)
                        if not pairs:
                            raise ValueError(f"No current {band} coverage for {name}")
                        stack_band(folder/"cropped", band, pairs=pairs)
                    stages["galex_process"] = "complete"
                if download_decals:
                    from decals_pipeline.query import DECaLSQuery
                    DECaLSQuery(output_dir=str(self.decals.download_dir), layer=self.decals.layer).query_region(
                        SkyCoord(row.ra_deg, row.dec_deg, unit="deg"), size=decals_size_pixels,
                        download_type="all", galid=name)
                    stages["decals_download"] = "complete"
                if split_decals:
                    self.decals.save_split_data(name)
                    stages["decals_split"] = "complete"
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
