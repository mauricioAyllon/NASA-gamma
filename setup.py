from setuptools import setup
import os

setup(
    name="nasagamma",
    use_scm_version={"write_to": "nasagamma/version.py"},
    description="Gamma spectroscopy tools",
    url="https://github.com/mauricioAyllon/NASA-gamma",
    author="Mauricio Ayllon Unzueta",
    author_email="mauri.ayllon12@gmail.com",
    packages=["nasagamma"],
    scripts=["gammaGUI-qt"],
    package_data={
        "nasagamma": [
            os.path.join("nasagamma", "data", "*txt"),
            os.path.join("nasagamma", "data", "*csv"),
            os.path.join("nasagamma", "*ui"),
        ]
    },
    install_requires=[
        "docopt >= 0.6.2",
        "lmfit > 1.0.2",
        "dateparser >= 1.1.1",
        "pandas > 1.4.2",
        "matplotlib == 3.6.3",
        "mplcursors >= 0.5.2",
        "pyarrow >= 11.0.0",
        "fastparquet >= 2023.4.0",
        "natsort >= 7.1.1",
        "plotly >= 5.22.0",
        "scipy >= 1.9.3",
    ],
)
