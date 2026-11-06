from .pipeline import process_galaxy, process_many
from .plotter import plot_fuv_nuv
from .download import download_galex
from .class_pipeline import GalexPipeline

__version__ = '0.2.0'
__all__ = ['process_galaxy', 'process_many', 'plot_fuv_nuv', 'download_galex', 'GalexPipeline']
