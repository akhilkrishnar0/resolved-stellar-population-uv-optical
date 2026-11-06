"""Full-field GALEX previews and explicit, non-destructive input selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle
from galex_pipeline.io_utils import unzip_galex_files, organize_fits


def load_exclusions(base_dir, overrides=None):
    folder = Path(base_dir)/"galex_review"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/"exclusions.json"
    if not path.exists():
        path.write_text("{}\n")
    mapping = json.loads(path.read_text())
    if not isinstance(mapping, dict):
        raise ValueError("exclusions.json must contain a JSON object")
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError("galex_exclude must map galaxy names to lists of intensity FITS filenames")
        mapping.update(overrides)
    if not isinstance(mapping, dict) or any(not isinstance(v, list) or any(not isinstance(n, str) or Path(n).name != n for n in v) for v in mapping.values()):
        raise ValueError("Exclusions must map galaxy names to lists of exact intensity FITS basenames")
    return {k.replace(" ", ""): v for k, v in mapping.items()}


def prepare_review(folder, output, galaxy, ra, dec, excluded=(), previews=True):
    folder, output = Path(folder), Path(output)
    unzip_galex_files(folder)
    organize_fits(folder)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for band in ("FUV", "NUV"):
        for intensity in sorted((folder/band/"int").glob("*-int.fits")):
            response = folder/band/"rrhr"/intensity.name.replace("-int.fits", "-rrhr.fits")
            rows.append(dict(galaxy=galaxy, band=band, filename=intensity.name,
                intensity_path=str(intensity), response_path=str(response),
                response_exists=response.is_file(), excluded=intensity.name in excluded,
                status="excluded" if intensity.name in excluded else "candidate",
                preview=str(output/band/(intensity.stem+".png")),
                ra_deg=ra, dec_deg=dec, preview_error=""))
    unknown = set(excluded)-{r["filename"] for r in rows}
    if unknown:
        raise ValueError(f"Unknown GALEX exclusion filenames for {galaxy}: {sorted(unknown)}")
    columns = ["galaxy", "band", "filename", "intensity_path", "response_path",
        "response_exists", "excluded", "status", "preview", "ra_deg", "dec_deg", "preview_error"]
    table = pd.DataFrame(rows, columns=columns)
    csv = output/"inputs.csv"
    table.to_csv(csv, index=False)
    if previews:
        for i, row in table.iterrows():
            try:
                plot_pair(row, SkyCoord(ra, dec, unit="deg"))
            except Exception as exc:
                table.at[i, "preview_error"] = f"{type(exc).__name__}: {exc}"
        table.to_csv(csv, index=False)
    return table


def plot_pair(row, center):
    fig = Figure(figsize=(12, 6), constrained_layout=True)
    FigureCanvasAgg(fig)
    for ax, key, label in zip(fig.subplots(1, 2), ("intensity_path", "response_path"), ("Intensity", "Relative response / exposure")):
        path = Path(row[key])
        ax.set_title(label)
        if not path.is_file():
            ax.text(.5, .5, "Missing FITS", transform=ax.transAxes, ha="center")
            continue
        with fits.open(path, memmap=True) as hdus:
            data, header = hdus[0].data, hdus[0].header
            if data is None or data.ndim != 2:
                raise ValueError(f"Expected 2D primary image: {path}")
            ny, nx = data.shape
            step = max(1, int(np.ceil(max(nx, ny)/1600)))
            display = np.asarray(data[::step, ::step], dtype=float)
            finite = display[np.isfinite(display)]
            lo, hi = np.percentile(finite, [1, 99.5]) if finite.size else (0, 1)
            if hi <= lo:
                hi = lo+1
            ax.imshow(display, origin="lower", cmap="gray", extent=(-.5, nx-.5, -.5, ny-.5),
                norm=ImageNormalize(vmin=lo, vmax=hi, stretch=AsinhStretch()))
            x, y = WCS(header).celestial.world_to_pixel(center)
            inside = np.isfinite([x, y]).all() and -.5 <= x < nx-.5 and -.5 <= y < ny-.5
            if inside:
                ax.add_patch(Circle((x, y), .025*max(nx, ny), fill=False, color="cyan", lw=1.5))
                ax.plot(x, y, "+", color="red", ms=8)
            else:
                ax.text(.03, .96, "Galaxy centre outside field", transform=ax.transAxes,
                    va="top", color="red", bbox=dict(facecolor="white", alpha=.8))
            ax.set(xlim=(-.5, nx-.5), ylim=(-.5, ny-.5), xlabel="Original pixel x", ylabel="Original pixel y")
    fig.suptitle(f"{row['galaxy']} | {row['band']} | {row['filename']}"+ (" | EXCLUDED" if row['excluded'] else ""), fontsize=10)
    output = Path(row["preview"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    fig.clear()
