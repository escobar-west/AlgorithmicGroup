import os
import sys
import functools
import traceback
import pdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithmicgroup import *

def debug_on(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            info = sys.exc_info()
            traceback.print_exception(*info)
            pdb.post_mortem(info[2])
            raise
    return wrapper 
