from distutils.core import setup, Extension
import numpy
import os
import re
import requests
import glob

# get NUMPY.I
np_version = re.compile(r'(?P<MAJOR>[0-9]+)\.'
                        '(?P<MINOR>[0-9]+)').search(numpy.__version__)
np_version_string = np_version.group()
np_version_info = {key: int(value)
                   for key, value in np_version.groupdict().items()}
np_file_name = 'numpy.i'
np_file_url = 'https://raw.githubusercontent.com/numpy/numpy/maintenance/' + \
              np_version_string + '.x/tools/swig/' + np_file_name
if(np_version_info['MAJOR'] == 1 and np_version_info['MINOR'] < 9):
    np_file_url = np_file_url.replace('tools', 'doc')
chunk_size = 8196
with open(np_file_name, 'wb') as file:
    for chunk in requests.get(np_file_url, stream=True).iter_content(chunk_size):
        file.write(chunk)

# build
os.environ['CC'] = 'g++'
# os.environ['CC'] = 'clang++'
setup(name='Goplanes', version='1.0',
      ext_modules=[Extension('_goplanes',
                   ['goplanes.cpp', 'goplanes.i'] + glob.glob('../src/*.cpp'),
                   extra_compile_args=["-Wno-deprecated", "-O3", "-std=c++11"],
                   include_dirs=[numpy.get_include(), '.', '../src'])
                   ]
      )
