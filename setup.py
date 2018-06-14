
from distutils.core import setup

import sys, platform
assert sys.version_info >= (3,3), "KGAN requires Python version >= 3.3. You have "+platform.python_version()

setup(
    name='kgan',
    version='0.1dev',
    py_modules=['kgan'],
    long_description=open('README.md').read(),
    
)