
from __future__ import print_function

import fnmatch
import glob
import os
import platform

def find_filenames(d, fnPattern):
    """
    Find all the filenames in directory d. A ascending sort is applied by default.

    d: The directory.
    fnPattern: The file pattern like "*.png".
    return: A list contains the strings of the sortted file names.
    """

    # Compose the search pattern.
    s = d + "/" + fnPattern

    fnList = glob.glob(s)
    fnList.sort()

    return fnList

def find_filenames_recursively(d, fnPattern):
    """
    Find all the filenames srecursively and starting at d. The resulting filenames
    will be stored in a list.

    d: The root directory.
    fnPattern: The file pattern like "*.json".
    return: A list contains the strings of the file names with relative paths.

    NOTE: This function heavily relys on the Python package glob. Particularly, the glob
    for Python version higher than 3.4.
    """

    # Test if the version of python is greater than 2
    if ( 2 < int(platform.python_version()[0]) ):
        # Compose the search pattern.
        s = d + "/**/" + fnPattern
    
        fnList = glob.glob(s, recursive = True)
    else:
        fnList = [ os.path.join(dirpath, f)
                    for dirpath, dirnames, files in os.walk(d)
                    for f in fnmatch.filter(files, fnPattern) ]

    fnList.sort()

    return fnList
