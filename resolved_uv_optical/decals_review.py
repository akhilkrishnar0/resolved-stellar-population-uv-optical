"""Review the downloaded DECaLS science cube without splitting or resampling."""
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle


def prepare_decals_review(download_dir, output, galaxy, ra, dec):
    download_dir, output = Path(download_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    image = download_dir/"fits"/f"{galaxy}.fits"
    products = {"science": image,
        "invvar": download_dir/"invvar"/f"{galaxy}_invvar.fits",
        "psf": download_dir/"psf"/f"{galaxy}_psf.fits"}
    pd.DataFrame([dict(galaxy=galaxy, product=k, path=str(v), exists=v.is_file())
        for k, v in products.items()]).to_csv(output/"files.csv", index=False)
    preview = output/"grz.png"
    preview.unlink(missing_ok=True)
    with fits.open(image) as hdus:
        data, header = hdus[0].data, hdus[0].header
        if data is None or data.ndim != 3 or data.shape[0] != 3:
            raise ValueError("Expected DECaLS primary science cube (3, ny, nx), ordered g/r/z")
        wcs = WCS(header).celestial
        x, y = wcs.world_to_pixel(SkyCoord(ra, dec, unit="deg"))
        ny, nx = data.shape[1:]
        inside = np.isfinite([x, y]).all() and -.5 <= x < nx-.5 and -.5 <= y < ny-.5
        fig = Figure(figsize=(15, 5), constrained_layout=True)
        FigureCanvasAgg(fig)
        for ax, band, plane in zip(fig.subplots(1, 3), ("g", "r", "z"), data):
            finite = plane[np.isfinite(plane)]
            lo, hi = np.percentile(finite, [1, 99.5]) if finite.size else (0, 1)
            if hi <= lo:
                hi = lo+1
            ax.imshow(plane, origin="lower", cmap="gray",
                norm=ImageNormalize(vmin=lo, vmax=hi, stretch=AsinhStretch()))
            if inside:
                ax.add_patch(Circle((x, y), .025*max(nx, ny), fill=False, color="cyan", lw=1.5))
                ax.plot(x, y, "+", color="red", ms=8)
            else:
                ax.text(.03, .95, "Galaxy centre outside field", transform=ax.transAxes,
                    va="top", color="red", bbox=dict(facecolor="white", alpha=.8))
            ax.set(title=f"{galaxy} | {band}", xlim=(-.5, nx-.5), ylim=(-.5, ny-.5))
        fig.savefig(preview, dpi=130)
        fig.clear()
    return str(preview)
