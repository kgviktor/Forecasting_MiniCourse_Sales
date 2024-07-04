from setuptools import setup, find_packages

setup(
    name='Forecasting_MiniCourse_Sales',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'numpy==2.0.0',
        'pandas==2.2.2',
        'matplotlib==3.9.0',
        'seaborn==0.13.2',
        'optuna==3.6.1',
        'lightgbm==4.4.0',
        'scikit-learn==1.5.0',
    ],
    extras_require={
        'dev': [
            'pytest==8.2.2',
        ],
    },
)