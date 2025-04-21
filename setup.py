from setuptools import setup, find_packages

setup(
    name="pdf_to_markdown",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "PyMuPDF==1.25.5",
        "python-dotenv==1.1.0",
        "requests==2.32.3",
    ],
    python_requires=">=3.8",
) 