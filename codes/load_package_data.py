# -*- coding: utf-8 -*-

import os, sys
import warnings
import pkg_resources

import numpy as np
from scipy.sparse.csgraph import laplacian


"""
This is the class to represent loading package data

Input to get file name
----------------------
file name : str
	file to be loaded
"""

class LoadPackageData():

	def __init__(self, filename):

		super(LoadPackageData, self).__init__()



	def getFileName(filename)

		data_path = pkg_resources.resource_filename(__name__, 'data')

		return data_path+'/'+filename

