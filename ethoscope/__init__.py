"""
Automatic workflow from accelerometer data to active/inactive states using Vectorial
Dynamic Body Acceleration (VeDBA).
"""

__version__ = "0.0.1 ('Alpha')"
__author__ = "Pranav Minasandra"
__license__ = "MIT"
__url__ = "nyet"

__all__ = ['parse']

from .ethoscope import parse

APP_NAME = "ethoscope"
APP_AUTHOR = __author__
