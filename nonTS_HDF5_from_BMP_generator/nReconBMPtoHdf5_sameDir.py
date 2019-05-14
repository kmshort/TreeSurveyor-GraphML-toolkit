# a simple script to convert an nRecon image sequence to an H5/hdf5 file; by Kieran Short. Monash University, Feb 22, 2019.

import os
import re
import h5py
import numpy as np
from PIL import Image


stackName = 'nReconStack'
dataset='CytoK'

def nReconBmpSequence_to_Hdf5():
    """
    Saves nRecon 8bit greyscale bmp image sequences as 4 channel HDF5 datsets (ZYXC)
    """
    dataPath = PATH = os.path.abspath('')
    fileNamePrefix = os.path.dirname(dataPath).split("\\")[-1]
    
    h5Filename = fileNamePrefix + '_' + dataset + '.h5'
    #lazy loading of the files from the same dir as the script from the same image directory at the moment
    PATH = dataPath
    
    # pluck out the BMP images that match the nRecon reconstructed slice data naming conventions
    images = [f for f in os.listdir(PATH) if re.search(r'.*_rec(\d).*(bmp)', f)]
    # make a numpy array of the images
    threeDarray = np.array([np.array(Image.open(PATH+'\\'+slice_image)) for slice_image in images])
    
    # add in a fourth dummy dimension to turn the zyx slices into zyxc -- c is just a 1d empty axis
    fourDarray = np.expand_dims(threeDarray, axis=3)
    
    # set up the hdf5 file for writing
    h5File = h5py.File(h5Filename, 'w')
    # add the ndarray of zyxc dimensions to the hdf5 file
    h5File.create_dataset(stackName,data=fourDarray)
    # hdf5 file written, so close it
    h5File.close()


def main():

    nReconBmpSequence_to_Hdf5()

main()