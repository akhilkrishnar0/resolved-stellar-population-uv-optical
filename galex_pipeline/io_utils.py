"""GALEX product pairing and local file organization."""
from pathlib import Path
import gzip
import re
import shutil

PRODUCT = re.compile(r"^(?P<tile>.+)-(?P<band>fd|nd)-(?P<kind>intbgsub|int|skybg|rrhr)\.fits(?:\.gz)?$")


def unzip_galex_files(folder):
    for source in Path(folder).glob("*.gz"):
        if not PRODUCT.match(source.name):
            continue
        target = source.with_suffix("")
        temporary = target.with_suffix(".fits.part")
        try:
            with gzip.open(source, "rb") as fin, temporary.open("wb") as fout:
                shutil.copyfileobj(fin, fout)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def organize_fits(folder):
    folder = Path(folder)
    for band in ("FUV", "NUV"):
        for kind in ("int", "rrhr", "skybg", "intbgsub"):
            (folder / band / kind).mkdir(parents=True, exist_ok=True)
    for source in folder.glob("*.fits"):
        match = PRODUCT.match(source.name)
        if match:
            band = "FUV" if match['band'] == "fd" else "NUV"
            source.replace(folder / band / match['kind'] / source.name)


def image_pairs(folder, band):
    """Require an exact observation/band match; never fall back to EXPTIME."""
    folder = Path(folder)
    pairs = []
    for intensity in sorted((folder / band / "int").glob("*-int.fits")):
        response = folder / band / "rrhr" / intensity.name.replace("-int.fits", "-rrhr.fits")
        if not response.is_file():
            raise FileNotFoundError(f"Missing response for {intensity.name}: {response}")
        pairs.append((intensity, response))
    return pairs
