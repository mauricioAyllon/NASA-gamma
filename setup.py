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
    scripts=["gammaGUI", "ecalGUI", "gammaGUI-qt"],
    package_data={
        "": [
            os.path.join("qt_gui.ui"),
        ]
    },
)
