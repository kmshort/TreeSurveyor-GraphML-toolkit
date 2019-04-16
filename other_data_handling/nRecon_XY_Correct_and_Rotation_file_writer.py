'''
OPT scan data correction by Kieran Short
Python2.7

This script helps recover dodgy OPT scans which may have had a rotation shift during scanning, where
(1) the sample axis of rotation is not perpendicular to the FOV captured,
and/or
(2) where the XY position has shifted during the scan.

You can enter in an angle to rotate the TIFF output from the OPT scan. Additionally, if you have used nRecon's SHIFT+ALT+CTRL "engineering mode" to already aline the 
original scan TIFF data to a reference "XY scan", this script can load the iterated XY-shift data from the .CSV that this function in nRecon generates.
Using this, it will load the shifted values written in the .CSV sheet by nRECON, and shift the OPT scan by that many pixels, producing a final dataset that is rotated,
and XY-shifted. The corrected OPT scan TIFF files are placed in a directory called /corrected. And for simplicity, the Bioptonics OPT3001 .log file is copied
to the /corrected directory to simplify loading the corrected data into nRecon for reconstruction afterward.
The idea being that the resulting data has a perpendicular axis of rotation which is vertically stable throughout all 360 degrees of frames.
'''

import glob
import pandas as pd
from PIL import Image
import os
import sys
from shutil import copyfile
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='OPT scan correction')
parser.add_argument('--shift', action='store_true', help='Whether to invoke XY correction shift stamping, 0 = no, 1 = yes')

parser.add_argument('--rotate', type=float, 
                    help='Rotate the images, rotation degree value as float')
                    
args = parser.parse_args()
                    # Optional positional argument
shift_option = args.shift
rotation_option = args.rotate
print shift_option

if shift_option is True:
    spreadsheetFiles = glob.glob('*.csv') # Use *.* if you're sure all are images

    imageNameBase = spreadsheetFiles[0][:-7]
    imageFileExtension = '.tif'
    os.mkdir('corrected')
    logFilename=imageNameBase+'.log'
    copyfile(logFilename, 'corrected/'+logFilename)

    for f in spreadsheetFiles:

        csvCorrectionSheet = pd.read_csv(f, sep=',', skiprows=3)
    
        for index, row in csvCorrectionSheet.iterrows():
            imageNumber = str(int(row[0])).zfill(4)
            imageFilename = imageNameBase+imageNumber+imageFileExtension
            image = Image.open(imageFilename)
            width, height = image.size
            shifted_image = Image.new("I;16", (width, height), color='White')
        
            shifted_image.paste(image, (int(row[1]), int(row[2])))
            shiftedFilename = str('corrected\\')+imageFilename
            
            if rotation_option is not None:
                shifted_image_rotated = shifted_image.rotate(-0.6)
                shifted_image_rotated.save(shiftedFilename)
                print 'Rotating XY shifted',imageFilename, 'by', rotation_option, 'degrees'
            else:
                shifted_image.save(shiftedFilename)
                print 'XY shifting', imageFilename


else:
    if rotation_option is not None:
        os.mkdir('corrected')
        logFiles = glob.glob('*.log') # Use *.* if you're sure all files are images
        copyfile(logFiles[0], 'corrected/'+logFilename)
        imageNameBase = logFiles[0][:-4]
        
        searchTerm = imageNameBase+('[0-9]' * 4)+'*.tif'
        
        imageFiles = glob.glob(searchTerm)
        
        for item in imageFiles:
            image = Image.open(item)
            image_rotated = image.rotate(rotation_option)
            rotatedFilename = str('corrected\\')+item
            image_rotated.save(rotatedFilename)
            print 'Rotating',item, 'by', rotation_option, 'degrees'
        
        #print imageFiles
    else:
        sys.exit("No options entered")
    