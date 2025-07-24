from setuptools import setup, find_packages

setup(
    name="evo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # System
        'python-dotenv',
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
        'pytest-mock>=3.10.0',
        'pytest-asyncio>=0.21.0',
        'psutil>=5.9.0',
        
        # Data Handling
        'numpy',
        'pandas',
        
        # Visualization
        'matplotlib',
        'jupyter',
        
        # Trading
        'alpaca-py',
        'backtrader',
        
        # Machine Learning
        'scikit-learn',
        'tensorflow>=2.9.0',
        'stable-baselines3[extra]',
        'gym',
    ],
    entry_points={
        'console_scripts': [
            'evo = evo.cli.evo:main',
        ],
    },
    include_package_data=True,
    description="EVO Trading System",
    author="Grant Morgan",
    author_email="grant.t.morgan@gmail.com",
    url="https://github.com/grant-tm/EVO",
) 