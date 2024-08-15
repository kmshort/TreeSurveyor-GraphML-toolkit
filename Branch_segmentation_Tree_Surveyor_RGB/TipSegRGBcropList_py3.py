import numpy as np
import argparse
import csv
import textwrap
import sys
sys.path.append('../GraphML_Routine_module/')
import tree_routines_X3_py3 as tx

def main():
    """Segments tips from Tree Surveyor Raw data volumes.
    Takes as input, a tree surveyor GraphML file. This contains all the branch names that correspond to
    the detected tips in the list provided. A raw volume is input which is the Tree Surveyor processed sample with cartesian
    coordinates which match up with the tips and their 3D coordiantes provided by the GraphML.
    """
    
    parser = argparse.ArgumentParser(prog='TipSeg', formatter_class=argparse.RawTextHelpFormatter, description=textwrap.dedent('''\
    TipSeg segments tips from Tree Surveyor raw data volumes.
    
    Three data files are required, in this order, and *MUST* correspond to the same analysed sample.
    
    1. A GraphML file exported from an analysed Tree Surveyor organ.
    
    2. A Raw volume. Exported from Tree Surveyor. Can be either
       
       FilteredSourceVolume.dat.raw     (greyscale 8bit volume)   (median filtered sample)
       VolumeFixData.dat.raw            (8bit thresholded volume) (thresholded filled segment ducts)
       
       This volume has the same cartesian coordinates and XYZ dimensions as the GraphML file.
    
    3. An RGB volume. Exported from Tree Surveyor. Can either be from Tree Surveyor (default) or from imageJ - if previously processed using it.
    
    You can optionally add an argument (default is TS):
    TS
    or
    imagej
    To cater for tree surveyor or imagej RGB volumes, which use different colour orders.
    
    Exports all listed branches/tips, registered so that they are facing "vertically", to a series of directories, in 8-bit RAW format
    
    '''))
    
    parser.add_argument('graph_file')
    parser.add_argument('raw8bit_file')
    parser.add_argument('raw_RGB_file')
    parser.add_argument('RGBraw_file_type', type=str, default='TS', const='TS', nargs="?", choices=['TS', 'imagej'])
    args = parser.parse_args()
    graphML = args.graph_file
    RGBrawFileType = args.RGBraw_file_type
    rawRGB = args.raw_RGB_file
    rawFilteredVolume = args.raw8bit_file
    
    resolution = 3.21
    
    tipList = ["1.2.2.2.2.2","1.2.2.2.2.1.1","1.1.1.2.1.2.2.2","2.1.2.2.2.1.2.2"]
    
    nxGraphML = tx.loadGraphML(graphML)
    TSFilteredVolume = tx.loadRaw8bitVolume(rawFilteredVolume, nxGraphML)
    TSRGBVolume = tx.loadRGBrawVolume(rawRGB, nxGraphML, RGBrawFileType)
    
    with open('coordinateData.csv', 'w', newline='') as csvfile:
        dataWriter = csv.writer(csvfile, delimiter =',')
        dataWriter.writerow(['BranchID','Dimensions','BaseCoordinate','EndCoordinate'])
        for tip in tipList:
            print("Extracting: " + str(tip))
            branchdata = []
            croppedRGBSegment = tx.cropSegment(nxGraphML,tip,np.copy(TSRGBVolume))
            noTree3Darray = tx.removeYellowVoxelTree(np.copy(croppedRGBSegment))
            hsv_Array = tx.rgbArray_2_hsvArray(np.copy(noTree3Darray))
            TSFilteredSegmentCrop = tx.cropSegment(nxGraphML,tip,np.copy(TSFilteredVolume))
            colourSampleCoordinates = tx.getBranchMiddleSplineControlPoint(nxGraphML, tip, cropped=True)
            segmentedArrayHSV = tx.segmentArray(np.copy(hsv_Array), colourSampleCoordinates)
            splineMask = tx.branchSplineSphereMask(hsv_Array, nxGraphML,tip, sizeBoost = 10, cropped=True)
            filteredMask = tx.ArrayMaskWhite(np.copy(segmentedArrayHSV),np.copy(splineMask))
            filteredTSSegment = tx.ArrayMaskGrey(np.copy(filteredMask), np.copy(TSFilteredSegmentCrop))
            eightBitfilteredTSSegment = filteredTSSegment.astype('uint8')
            rotatedTSSegment = tx.rotateVertical(eightBitfilteredTSSegment, nxGraphML, tip)
            autocropped = tx.autoCrop3d(rotatedTSSegment)
        
            #print "rotated tip size: "+str(autocropped.shape)
            tx.export_array_8bit_byte_raw(autocropped, str(tip+".raw"))
        
            endCoordinate, baseCoordinate = tx.calculateTransformedCoordinates(nxGraphML, tip, autocropped)
            #print newCoordinate1,newCoordinate2
        
            dimensions = "(" + str(autocropped.shape[0]) + "," + str(autocropped.shape[1]) + "," + str(autocropped.shape[2]) + ")"
            formattedBaseCoordinate = "(" + str(baseCoordinate[0]) + "," + str(baseCoordinate[1]) + "," + str(baseCoordinate[2]) + ")"
            formattedEndCoordinate = "(" + str(endCoordinate[0]) + "," + str(endCoordinate[1]) + "," + str(endCoordinate[2]) + ")"
        
            branchdata = [tip, dimensions, formattedBaseCoordinate, formattedEndCoordinate]
            dataWriter.writerow(branchdata)
            
            # write raw to minc converter line for Mattijs
            # convert um dimensions to millimetre, used by rawtominc in linux
            resolutionMinc = resolution / 1000
    
            with open("convert_raw_to_minc.txt", "a") as myfile:
                myfile.write("rawtominc -2 -unsigned -byte -real_range 0 255 -xstep " + str(resolutionMinc) + " -ystep -" + str(resolutionMinc) + " -zstep " + str(resolutionMinc) + " -xstart 0 -ystart 0 -zstart 0 " + str(tip) + ".mnc " + str(autocropped.shape[0]) + " " +str(autocropped.shape[1]) + " " + str(autocropped.shape[2]) + "\n")

main()