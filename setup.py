from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="xarrayvideo",
    version="0.1.0",
    author="Oscar J. Pellicer-Valero",
    author_email="oscar.pellicer@uv.es",
    description="Save multichannel data from xarray datasets as videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OscarPellicer/xarrayvideo",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "xarray",
        "numpy",
        "scikit-image",
        "scikit-learn",
        "pyyaml",
        "zarr",
        "netcdf4",
        "ffmpeg-python",
        "gcsfs",
        "pillow",
        "tqdm",
        "seaborn",
        "h5netcdf",
        "tacoreader",
        "pytortilla",
        "tacotoolbox",
    ],
    extras_require={
        "satalign": ["satalign"],
        "plotting": ["ipython", "opencv-python", "txyvis"],
        "metrics": ["torchmetrics"],
        "gdal": ["gdal"],
        "all": ["satalign", "ipython", "opencv-python", "txyvis", "torchmetrics"],
    },
)