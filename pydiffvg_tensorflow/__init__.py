import tensorflow as tf
try:
    import diffvg
except ImportError:
    print("Warning: diffvg is not installed when you import pydiffvg_tensorflow.")
from .device import *
from .shape import *
from .pixel_filter import *
from .render_tensorflow import *
from .image import *
from .color import *
import os.path

print(os.path.dirname(diffvg.__file__))

if tf.__cxx11_abi_flag__ == 0:
    __data_ptr_module = tf.load_op_library(os.path.join(os.path.dirname(diffvg.__file__), 'libdiffvg_tf_data_ptr_no_cxx11_abi.so'))
else:
    assert(tf.__cxx11_abi_flag__ == 1)
    __data_ptr_module = tf.load_op_library(os.path.join(os.path.dirname(diffvg.__file__), 'libdiffvg_tf_data_ptr_cxx11_abi.so'))

def data_ptr(tensor):    
    addr_as_uint64 = __data_ptr_module.data_ptr(tensor)
    return int(addr_as_uint64)
