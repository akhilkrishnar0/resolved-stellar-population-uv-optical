"""Convert manual galaxy centres into validated pipeline input rows."""
import re
import numpy as np
import pandas as pd
def prepare_cutout_table(galaxies,distances,sizes,centers):
    if not (len(galaxies)==len(distances)==len(sizes)):
        raise ValueError("galaxies, distances and sizes must have equal lengths")
    if hasattr(centers,"to_pandas"):
        centers=centers.to_pandas()
    if not isinstance(centers,pd.DataFrame):
        raise TypeError("centers must be a pandas DataFrame or Astropy Table")
    rename={"Galaxy":"galaxy","RA":"ra_deg","DEC":"dec_deg"}
    centers=centers.rename(columns=rename).copy()
    for col in ("galaxy","ra_deg","dec_deg"):
        if col not in centers:
            raise ValueError(f"centers missing column: {col}")
    def key(name):
        return re.sub(r"[\s-]+","",str(name)).upper()
    centers["key"]=centers["galaxy"].map(key)
    if centers["key"].duplicated().any():
        raise ValueError(f"Duplicate normalized centre names: {centers.loc[centers['key'].duplicated(),'galaxy'].tolist()}")
    run=pd.DataFrame({"galaxy":galaxies,"distance":distances,"size":sizes})
    run["key"]=run["galaxy"].map(key)
    run=run.merge(centers[["key","ra_deg","dec_deg"]],on="key",how="left",validate="many_to_one").drop(columns="key")
    for c in ("ra_deg","dec_deg","distance","size"):
        run[c]=pd.to_numeric(run[c],errors="coerce")
    missing=run.loc[run[["ra_deg","dec_deg"]].isna().any(axis=1),"galaxy"].tolist()
    if missing:
        raise ValueError(f"Missing coordinates: {missing}")
    if not (np.isfinite(run[["ra_deg","dec_deg","distance","size"]].to_numpy()).all() and run["ra_deg"].between(0,360,inclusive="left").all() and run["dec_deg"].between(-90,90).all() and (run["distance"]>0).all() and (run["size"]>0).all()):
        raise ValueError("Invalid coordinates, distance or cutout size")
    return run
