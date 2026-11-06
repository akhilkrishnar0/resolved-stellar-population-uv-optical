"""Standalone PSF matching for previously generated GALEX/DECaLS FITS cutouts."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.visualization import simple_norm
DEFAULT_FWHM = {"g":1.3,"r":1.2,"z":1.1,"FUV":4.2,"NUV":5.3}
def pixel_scale_arcsec(header):
    scales = np.asarray([s.to_value(u.arcsec) if hasattr(s,"to_value") else float(s)*3600.0 for s in WCS(header).celestial.proj_plane_pixel_scales()],dtype=float)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Invalid celestial WCS pixel scale")
    if abs(scales[0]-scales[1])/np.mean(scales) > 0.01:
        raise ValueError("Anisotropic pixel scales: circular pixel Gaussian is unsuitable")
    return float(np.mean(scales))
def nan_safe_convolve(data,kernel,min_weight=1e-6):
    valid = np.isfinite(data)
    if not valid.any():
        return np.full_like(data,np.nan,dtype=float)
    numerator = convolve_fft(np.where(valid,data,0.0),kernel,boundary="fill",fill_value=0.0,normalize_kernel=True,allow_huge=True)
    denominator = convolve_fft(valid.astype(float),kernel,boundary="fill",fill_value=0.0,normalize_kernel=True,allow_huge=True)
    return np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>min_weight)
class PSFMatchingPipeline:
    """Gaussian PSF approximation; processes cutouts/<galaxy>/<band>.fits."""
    def __init__(self,base_dir="cutouts",target_fwhm=5.3,native_fwhm=None):
        self.base_dir=Path(base_dir)
        self.target_fwhm=float(target_fwhm)
        self.native_fwhm=dict(DEFAULT_FWHM if native_fwhm is None else native_fwhm)
        if not np.isfinite(self.target_fwhm) or self.target_fwhm<=0:
            raise ValueError("target_fwhm must be positive")
    def process_galaxy(self,galaxy,make_plot=True):
        galaxy=str(galaxy).strip().replace(" ","")
        folder=self.base_dir/galaxy
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        outputs={}
        for band,native in self.native_fwhm.items():
            infile=folder/f"{band}.fits"
            if not infile.exists():
                print(f"Missing {band}: {infile}")
                continue
            native=float(native)
            if not np.isfinite(native) or native<=0 or native>self.target_fwhm+1e-9:
                raise ValueError(f"{band}: native FWHM {native} exceeds target {self.target_fwhm} or is invalid")
            with fits.open(infile) as hdul:
                data=np.asarray(hdul[0].data,dtype=float)
                header=hdul[0].header.copy()
            if data.ndim!=2:
                raise ValueError(f"{infile}: expected 2-D image")
            pixscale=pixel_scale_arcsec(header)
            sigma_kernel=np.sqrt(max(self.target_fwhm**2-native**2,0.0))/2.354820045/pixscale
            result=nan_safe_convolve(data,Gaussian2DKernel(sigma_kernel)) if sigma_kernel>1e-6 else data.copy()
            header["PSFMATCH"]=(True,"Approximate Gaussian PSF matching")
            header["PSFTARG"]=(self.target_fwhm,"Target FWHM, arcsec")
            header["PSFNAT"]=(native,"Assumed native FWHM, arcsec")
            header["PSFSIGPX"]=(float(sigma_kernel),"Gaussian kernel sigma, pixels")
            header.add_history("Gaussian PSF matching; NaN-aware normalized convolution")
            outfile=folder/f"{band}_psfmatch.fits"
            fits.writeto(outfile,result,header,overwrite=True)
            outputs[band]=outfile
            print(f"{galaxy} {band}: {outfile} (sigma={sigma_kernel:.3f} pixels)")
        if make_plot and outputs:
            fig,axes=plt.subplots(1,len(outputs),figsize=(4*len(outputs),4),squeeze=False)
            for ax,(band,path) in zip(axes[0],outputs.items()):
                data=fits.getdata(path).astype(float)
                if np.isfinite(data).any():
                    im=ax.imshow(data,origin="lower",cmap="gray",norm=simple_norm(data[np.isfinite(data)],"sqrt",percent=99))
                    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
                ax.set_title(f"{band}: {self.target_fwhm:g} arcsec")
                ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle(galaxy)
            fig.tight_layout()
            fig.savefig(folder/f"{galaxy}_psfmatched.png",dpi=150)
            plt.close(fig)
        return outputs
    def run(self,galaxies,make_plot=True):
        if isinstance(galaxies,str):
            galaxies=[galaxies]
        return {galaxy:self.process_galaxy(galaxy,make_plot=make_plot) for galaxy in galaxies}
def process_galaxy(galaxy,base_dir="cutouts",make_plot=True):
    return PSFMatchingPipeline(base_dir=base_dir).process_galaxy(galaxy,make_plot=make_plot)
