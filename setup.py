import setuptools

setuptools.setup(
    name="query-verse",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)