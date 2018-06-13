
from distutils.core import setup

import sys, platform
assert sys.version_info >= (3,3), "KGAN requires Python version >= 3.3. You have "+platform.python_version()

setup(
    name='KGan',
    version='0.1dev',
    packages=[],
    long_description=open('README.md').read(),
    
)