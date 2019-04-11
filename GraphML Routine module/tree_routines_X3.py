#tree_extraction_routines.py
#python2.7

import networkx as nx
import math
import ast
import sys
import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage.interpolation as interpolation
import scipy.interpolate as si
import StringIO
from itertools import chain, izip
from scipy.spatial import ConvexHull
from scipy.spatial import HalfspaceIntersection
import scipy as sp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors


# Updated to work with networkX 2.X -- many features of 1.X are broken by the upgrade. At the moment nx.edge has been replaced by nx.adj -- which should work with both 1.X and 2.X.  nx.node still works for now, but will apparently be broken in the future in networkX 3.X. 
# see  https://networkx.github.io/documentation/stable/release/migration_guide_from_1.x_to_2.0.html







"""A BUNCH OF FUNCTIONS FOR EXTRACTING DATA FROM TREE SURVEYOR GRAPHML DATA"""

def getVolumeDimensions(nxTreeSurveyorGraph):
    #get the dimensions of the nx graph object from a Tree Surveyor GraphML file with keys Grid Size XYZ.
    dimensions = (nxTreeSurveyorGraph.graph['Grid Size X'], nxTreeSurveyorGraph.graph['Grid Size Y'], nxTreeSurveyorGraph.graph['Grid Size Z'])
    #Return as a TUPLE in X, Y, Z order
    return(dimensions)

def getBranchKeyInfo(nxTreeSurveyorGraph, TreeSurveyorBranchID):
    # RETURNS NOTHING, JUST PRINTS TO SCREEN A BUNCh OF INFO ABOUT A BRANCH
    if TreeSurveyorBranchID == 'Root':
        print 'Requested TreeSurveyorBranchID is Root, has no quantified segment information, only position of ureter and angle of first branchpoint'
        print nxTreeSurveyorGraph.node['Root']
    elif "Extension Of " in TreeSurveyorBranchID:
        # If the segment contains "Extension Of", it implies it is a tip and the data is attached to the previous edge/node.
        name = TreeSurveyorBranchID[13:]
        tipsegment = name[:-2]
        print nxTreeSurveyorGraph.node[tipsegment]
        print 'Tip boundary location:', nxTreeSurveyorGraph.node[name]
        print nxTreeSurveyorGraph.adj[tipsegment][name]
    elif len(TreeSurveyorBranchID) == 1:
        print nxTreeSurveyorGraph.node[TreeSurveyorBranchID]
        print nxTreeSurveyorGraph.adj['Root'][TreeSurveyorBranchID]
    else:
        parent = TreeSurveyorBranchID[:-2]
        print nxTreeSurveyorGraph.node[TreeSurveyorBranchID]
        print nxTreeSurveyorGraph.adj[parent][TreeSurveyorBranchID]
    
def getTipIDs(nxTreeSurveyorGraph):
    # supply a networkX graph (from networkX library) from TreeSurveyor, will return a list of all branch IDs that are TIPS.
    tips = []
    for l in nxTreeSurveyorGraph.node:
        if "Extension Of " in l:
            tipID = l[13:]
            tips.append(tipID)
    #Return a LIST
    return tips

def getBranchPointTotal(nxTreeSurveyorGraph):
    branchNodeTotal = nxTreeSurveyorGraph.number_of_nodes() -1
    return branchNodeTotal
    
    
def getAllCoordinates(nxTreeSurveyorGraph):
    '''#getsAllCoordinates, returns them as a numpy array... can also return as a list, it's there if you want to change it
    '''
    coordinateArrayList = []
    for branchID in nxTreeSurveyorGraph.node:
        if not branchID == 'Root':
            if not branchID == 'unnamed':
        # supply a networkX graph (from networkX library) from TreeSurveyor and a Branch ID as STRING, will return the coordinates of the END node of the branch
                xCoord = nxTreeSurveyorGraph.node[branchID]['Position X']
                yCoord = nxTreeSurveyorGraph.node[branchID]['Position Y']
                zCoord = nxTreeSurveyorGraph.node[branchID]['Position Z']
    
                coordinates = [xCoord,yCoord,zCoord]
                coordinateArrayList.append(coordinates)
            #print coordinates
    
        coordinateArray = np.asarray(coordinateArrayList)
    
    #print coordinateArrayList
    #print CoordinateArray

    return coordinateArray

def getKidneyHullVolume(nxTreeSurveyorGraph):
    ''' Gets the convex hull volume of all the points in the tree surveyor graphML file, using a convex Hull approach.
    WARNING, this is based on a 1x1x1 grid... it DOES NOT account for voxel-to-real world size.
    Check your pixel size. Could be 3.21 or otherwise.. so final actual volume would be volume*3.21*3.21*3.21 (um^3)
    This routine DOES NOT calculate that'''
    allCoordinates = getAllCoordinates(nxTreeSurveyorGraph)
    volume = ConvexHull(allCoordinates).volume
    return volume
    
def getKidneyHullSurfaceArea(nxTreeSurveyorGraph):
    ''' Gets the surface area of all the points in the tree surveyor graphML file, using a convex Hull approach.
    WARNING, this is based on a 1x1x1 grid... it DOES NOT account for voxel-to-real world size.
    Check your pixel size. Could be 3.21... therefore area needs to be area*3.21*3.21, and volume*3.21*3.21*3.21 (um^3)'''
    allCoordinates = getAllCoordinates(nxTreeSurveyorGraph)
    area = ConvexHull(allCoordinates).area
    return area

def plotKidneyHull(nxTreeSurveyorGraph):
    verts = getAllCoordinates(nxTreeSurveyorGraph)
    
    maxCoordinateDimensions = np.amax(verts, axis = 0)
    
    hull = ConvexHull(verts)
    faces = hull.simplices
    
    ax = a3.Axes3D(plt.figure())
    ax.dist=10
    ax.azim=30
    ax.elev=10
    ax.set_xlim([0,maxCoordinateDimensions[0]])
    ax.set_ylim([0,maxCoordinateDimensions[1]])
    ax.set_zlim([0,maxCoordinateDimensions[2]])
    
    for s in faces:
        sq = [
            [verts[s[0], 0], verts[s[0], 1], verts[s[0], 2]],
            [verts[s[1], 0], verts[s[1], 1], verts[s[1], 2]],
            [verts[s[2], 0], verts[s[2], 1], verts[s[2], 2]]
        ]
    
        f = a3.art3d.Poly3DCollection([sq])
        f.set_color(colors.rgb2hex(sp.rand(3)))
        f.set_edgecolor('k')
        f.set_alpha(0.1)
        ax.add_collection3d(f)

    plt.show()


    
    
def getBranchLength(nxTreeSurveyorGraph, TreeSurveyorBranchID):
    # supply a networkX graph (from networkX library) from TreeSurveyor and a Branch ID, will return the branch length
    if "Extension Of " in TreeSurveyorBranchID:
        TreeSurveyorBranchID = TreeSurveyorBranchID[13:]
        parentSegment = TreeSurveyorBranchID[:-2]
    elif len(TreeSurveyorBranchID) == 1:
        parentSegment = 'Root'
    else:
        parentSegment = TreeSurveyorBranchID[:-2]
        
    branchLength = nxTreeSurveyorGraph.adj[parentSegment][TreeSurveyorBranchID]['Length (microns)']
    # Return a FLOAT
    return branchLength

def getBranchDiameter(nxTreeSurveyorGraph, TreeSurveyorBranchID):
    # supply a networkX graph (from networkX library) from TreeSurveyor and a Branch ID, will return the branch diameter
    if "Extension Of " in TreeSurveyorBranchID:
        TreeSurveyorBranchID = TreeSurveyorBranchID[13:]
        parentSegment = TreeSurveyorBranchID[:-2]
    elif len(TreeSurveyorBranchID) == 1:
        parentSegment = 'Root'
    else:
        parentSegment = TreeSurveyorBranchID[:-2]
        
    branchDiameter = nxTreeSurveyorGraph.adj[parentSegment][TreeSurveyorBranchID]['Median diameter (microns)']
    # Return a FLOAT
    return branchDiameter

def getBranchVolume(nxTreeSurveyorGraph, TreeSurveyorBranchID):
    # supply a networkX graph (from networkX library) from TreeSurveyor and a Branch ID, will return the branch diameter
    if "Extension Of " in TreeSurveyorBranchID:
        TreeSurveyorBranchID = TreeSurveyorBranchID[13:]
        parentSegment = TreeSurveyorBranchID[:-2]
    elif len(TreeSurveyorBranchID) == 1:
        parentSegment = 'Root'
    else:
        parentSegment = TreeSurveyorBranchID[:-2]
        
    branchVolume = nxTreeSurveyorGraph.adj[parentSegment][TreeSurveyorBranchID]['Volume (cubic microns)']
    # Return a FLOAT
    return branchVolume
    
def getLocalBifurcationAngle(nxTreeSurveyorGraph, TreeSurveyorBranchID):
    # supply a networkX graph (from networkX library) from TreeSurveyor and a Branch ID, will return the branch diameter
    try: 
        branchLocalBifurcationAngle = nxTreeSurveyorGraph.node[TreeSurveyorBranchID]['Local Bifurcation Angle (degrees)']
    except:
        return
    # Return a FLOAT
    return branchLocalBifurcationAngle

def getGlobalBifurcationAngle(nxTreeSurveyorGraph, TreeSurveyorBranchID):
    # supply a networkX graph (from networkX library) from TreeSurveyor and a Branch ID, will return the branch diameter
    try:
        branchGlobalBifurcationAngle = nxTreeSurveyorGraph.node[TreeSurveyorBranchID]['Global Bifurcation Angle (degrees)']
    except:
        return
    # Return a FLOAT
    return branchGlobalBifurcationAngle   

def getLocalDihedralAngle(nxTreeSurveyorGraph, TreeSurveyorBranchID):
    # supply a networkX graph (from networkX library) from TreeSurveyor and a Branch ID, will return the branch diameter
    try:
        branchLocalDihedralAngle = nxTreeSurveyorGraph.node[TreeSurveyorBranchID]['Local Dihedral (p-)Angle (degrees)']
    except:
        return
    # Return a FLOAT
    return branchLocalDihedralAngle

def getGlobalDihedralAngle(nxTreeSurveyorGraph, TreeSurveyorBranchID):
    # supply a networkX graph (from networkX library) from TreeSurveyor and a Branch ID, will return the branch diameter
    try:
        branchGlobalDihedralAngle = nxTreeSurveyorGraph.node[TreeSurveyorBranchID]['Global Dihedral (p-)Angle (degrees)']
    except:
        return
    # Return a FLOAT
    return branchGlobalDihedralAngle  
    
def getBaseCoordinate(nxTreeSurveyorGraph, TreeSurveyorBranchID):
    # supply a networkX graph (from networkX library) from TreeSurveyor and a Branch ID, will return the coordinates of the START node of the branch
    if "Extension Of " in TreeSurveyorBranchID:
        parentSegment = TreeSurveyorBranchID[13:]
    elif len(TreeSurveyorBranchID) == 1:
        parentSegment = 'Root'
    else:
        parentSegment = TreeSurveyorBranchID[:-2]
    xCoord = nxTreeSurveyorGraph.node[parentSegment]['Position X']
    yCoord = nxTreeSurveyorGraph.node[parentSegment]['Position Y']
    zCoord = nxTreeSurveyorGraph.node[parentSegment]['Position Z']
    
    coordinates = [xCoord,yCoord,zCoord]

    # Return a LIST
    return coordinates

def getEndCoordinate(nxTreeSurveyorGraph, TreeSurveyorBranchID):
    # supply a networkX graph (from networkX library) from TreeSurveyor and a Branch ID as STRING, will return the coordinates of the END node of the branch
    xCoord = nxTreeSurveyorGraph.node[TreeSurveyorBranchID]['Position X']
    yCoord = nxTreeSurveyorGraph.node[TreeSurveyorBranchID]['Position Y']
    zCoord = nxTreeSurveyorGraph.node[TreeSurveyorBranchID]['Position Z']
    
    coordinates = [xCoord,yCoord,zCoord]
    # Return a LIST
    return coordinates    
    
def getBranchSplineControlPoints(nxTreeSurveyorGraph, TreeSurveyorBranchID,cropped=False):
    # supply a networkX graph (from networkX library) from TreeSurveyor and a Branch ID, will return the spline control points
    if "Extension Of " in TreeSurveyorBranchID:
        TreeSurveyorBranchID = TreeSurveyorBranchID[13:]
        parentSegment = TreeSurveyorBranchID[:-2]
    elif len(TreeSurveyorBranchID) == 1:
        parentSegment = 'Root'
    else:
        parentSegment = TreeSurveyorBranchID[:-2]
        
    splinePoints = nxTreeSurveyorGraph.adj[parentSegment][TreeSurveyorBranchID]['Spline Control Points']
    # Convert from UniCode string to list
    
    if cropped==True:
        cropOffset = findCropOffset(nxTreeSurveyorGraph, TreeSurveyorBranchID)
        splineControlPoints = list(ast.literal_eval(str(splinePoints)))
        
        splineControlPointsX = [int(item[0])-cropOffset[0] for item in splineControlPoints]
        splineControlPointsY = [int(item[1])-cropOffset[1] for item in splineControlPoints]
        splineControlPointsZ = [int(item[2])-cropOffset[2] for item in splineControlPoints]
        # print splineControlPoints
        
        combolist= list(chain.from_iterable(izip(splineControlPointsX, splineControlPointsY,splineControlPointsZ)))
        i=0
        splineControlPoints=[]
        while i<len(combolist):
            splineControlPoints.append(tuple(combolist[i:i+3]))
            i+=3

    else:
        splineControlPoints = list(ast.literal_eval(str(splinePoints)))
    
    
    #print splineControlPoints
    # Return LIST
    return splineControlPoints

def getBranchMiddleSplineControlPoint(nxTreeSurveyorGraph, TreeSurveyorBranchID,cropped=False):
    # supply a networkX graph (from networkX library) from TreeSurveyor and a Branch ID, will return the spline control points
    
    if cropped==True:
        splineControlPoints = getBranchSplineControlPoints(nxTreeSurveyorGraph, TreeSurveyorBranchID,cropped=True)
    else:
        splineControlPoints = getBranchSplineControlPoints(nxTreeSurveyorGraph, TreeSurveyorBranchID)
    
    splineCoordinateArray = bspline(splineControlPoints)
    splineCoordinateList = np.ndarray.tolist(splineCoordinateArray)
    splineCoordinateIntList  = [[int(j) for j in i]  for i in splineCoordinateList]

    middleControlPoint = list(findMiddle(splineCoordinateIntList))
    
    return middleControlPoint
    
def findMiddle(input_list):
    middle = float(len(input_list))/2
    if middle % 2 != 0:
        return input_list[int(middle - .5)]
    else:
        #just grab whatever's close to the middle
        return input_list[int(middle)]
    
def getBranchNeighbourID(nxTreeSurveyorGraph, TreeSurveyorBranchID):
    """ identifies neighbour BranchID(s) """
    # get the ID of the parent first.
    if TreeSurveyorBranchID == 'Root':
        print 'Requested node is Root, so has no neighbour'
        exit()
    if "Extension Of " in TreeSurveyorBranchID:
        TreeSurveyorBranchID = TreeSurveyorBranchID[13:]
        segment = TreeSurveyorBranchID[:-2]
        
    elif len(TreeSurveyorBranchID) == 1:
        segment = 'Root'
    else:
        segment = TreeSurveyorBranchID[:-2]
    
    # grab the children of the segment
    branches = nxTreeSurveyorGraph.adj[segment].keys()
    # remove the original branchID
    branches.remove(TreeSurveyorBranchID)

    if len(branches) > 1:
        print 'WARNING Flag: the getNeighbourID() function reports there are more than one sibling, in fact there are', len(branches), 'siblings. These involve', branches
    # Returns a STRING, could change this to return all siblings. At the moment, i'm just going with bifurcation
    return branches[0]
    
def getNeighbourData(nxTreeSurveyorGraph, TreeSurveyorBranchID, measure):
    neighbourID = getBranchNeighbourID(nxTreeSurveyorGraph, TreeSurveyorBranchID)
    if len(TreeSurveyorBranchID) == 1:
        parentSegment = 'Root'
    else:
        parentSegment = TreeSurveyorBranchID[:-2]

        validMeasures = ['length','diameter','volume','localBifurcation','globalBifurcation','localDihedral','globalDihedral','endCoordinate']

        functions = [getBranchLength,getBranchDiameter,getBranchVolume,getLocalBifurcationAngle,getGlobalBifurcationAngle,getLocalDihedralAngle,getGlobalDihedralAngle,getEndCoordinate]
    
    if measure in validMeasures:
        dataToGrab = validMeasures.index(measure)
        measuredData = functions[dataToGrab](nxTreeSurveyorGraph,neighbourID)
    else:
        print "\nError, ''"+str(measure)+"'' is not a valid measurement to return, please use any of the following:\nlength\ndiameter\nvolume\nlocalBifurcation\nglobalBifurcation\nlocalDihedral\nglobalDihedral\nendCoordinate\n"
        sys.exit()
    
    #RETURNS A FLOAT, or LIST depending on what the requested data is.
    return measuredData

def getBranchParent(nxTreeSurveyorGraph, TreeSurveyorBranchID):
    # there are usually only single alphanumeric characters separating generations in TreeSurveyor branchID naming. Therefore, removing two characters off the string will identify the branchID of the parent
    parentSegment = TreeSurveyorBranchID[:-2]
    return parentSegment
    
def getBranchGrandParent(nxTreeSurveyorGraph, TreeSurveyorBranchID):
    # there are usually only single alphanumeric characters separating generations in TreeSurveyor branchID naming. Therefore, removing four characters off the string will identify the branchID of the grandparent
    parentSegment = TreeSurveyorBranchID[:-2]
    grandParentSegment = TreeSurveyorBranchID[:-4]
    return grandParentSegment

def triTipTest(nxTreeSurveyorGraph, TreeSurveyorBranchID, tipsList):
    # Test to see if a branch (assumed to be a tip) has a parent whose sibling is a tip. By deduction, that makes all three a part of a tri-tip. This returns the BranchID of this third member.
    parentSegment = TreeSurveyorBranchID[:-2]
    grandParentSegment = TreeSurveyorBranchID[:-4]
    
    children = []
    for item in nxTreeSurveyorGraph.adj[grandParentSegment]:
        children.append(item)
    children.remove(parentSegment)

    tipsList = getTipIDs(nxTreeSurveyorGraph)
    
    if children[0] in tipsList:
        #print branchID, "might be part of a tri-tip"
        return children[0]
    return

def tipNubEstimator_crude_edition(nxTreeSurveyorGraph):
    
    '''
    This function generates a CSV spreadsheet, listing all BranchIDs for the tips in column 1. In column 2, the the number of sibling tip's derivative branches.
    after generation of the CSV, to find the likelyhood of internal tree "nubs", sort the CSV manually by column 2 from largest to smallest value.
    in Tree Surveyor, open the Kideny Segment Curation "Modify" tool, select "Yes" for showing the corrected segment data visualisation. Then when the
    window is open, hit the "Home" key on your keyboard, and enter the BranchID from the CSV sheet corresponding to the top hits of the derivatives.
    edit the tree as necessary. Eventually you'll get to a point in the list when the BranchIDs you're entering in are actual tips. I'd maybe check a
    couple more, and then stop.
    This function doesn't specifically isolate nubs, it just shows all derivatives.. hence I call it "crude_edition". 
    '''
    tipList = getTipIDs(nxTreeSurveyorGraph)
    

    
    with open('nubData.csv', 'wb') as csvfile:
        dataWriter = csv.writer(csvfile, delimiter =',')
        dataWriter.writerow(['BranchID','Neighbour Derivatives'])
        for tip in tipList:
            neighbourTip = tR.getBranchNeighbourID(nxGraphML, tip)
            derivativeTips = [i for i, s in enumerate(tipList) if neighbourTip in s]
            
            t = str(len(derivativeTips))
        
            branchdata = [tip, t]
            dataWriter.writerow(branchdata)
 
 
 
 
    
""" A BUNCH OF FUNCTIONS FOR MANIPULATING AND MEASURING 3D POINT DATA IN NUMPY ARRAYS"""
def distance3D(PointA, PointB, realWorldDistancePerPixel=3.21):
    # Calculates the euclidean distance between two points; each a LIST in formay [X,Y,Z]. Requires manual insertion of alternative distance. Most Kidney analysis on OPT3001 is 3.21um per pixel. So this is default. BEWARE IF YOUR SCALE IS DIFFERENT!!!!
    distance = math.sqrt((PointB[0] - PointA[0])**2 + (PointB[1] - PointA[1])**2 + (PointB[2] - PointA[2])**2)
    #RETURNS A FLOAT
    return distance*realWorldDistancePerPixel

def AngleBetweenTwoPoints(p1, p2):
    p1_unitVector = p1 / np.linalg.norm(p1)
    p2_unitVector = p2 / np.linalg.norm(p2)
    #returns an angle in radians
    return np.arccos(np.clip(np.dot(p1_unitVector, p2_unitVector), -1.0, 1.0))
    
def get_rotation_matrix(pointsVector, unitVector=None):
    # https://stackoverflow.com/questions/43507491/imprecision-with-rotation-matrix-to-align-a-vector-to-an-axis
    # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
    if unitVector is None:
        unitVector = [0.0, 1.0, 0.0]
    # Normalize vector length
    pointsVector = pointsVector / np.linalg.norm(pointsVector)
      
    # Get axis
    uvw = np.cross(unitVector,pointsVector)

    
    # compute trig values - no need to go through arccos and back
    rcos = np.dot(pointsVector, unitVector)
    rsin = np.linalg.norm(uvw)

    #normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw = uvw / rsin
    u, v, w = uvw
   
    # Compute rotation matrix - re-expressed to show structure
    rotationMatrix = (
        rcos * np.eye(3) +
        rsin * np.array([
            [ 0,  w, -v],
            [-w,  0,  u],
            [ v, -u,  0]
        ]) +
        (1.0 - rcos) * uvw[:,None] * uvw[None,:]
    )
    return rotationMatrix

    
def rotateVertical(volumeArray, nxTreeSurveyorGraph, TreeSurveyorBranchID, type=None, order=None):
    offsetStartCoordinate, offsetEndCoordinate = findCropPointOffset(nxTreeSurveyorGraph, TreeSurveyorBranchID)
    # make sure the input array is uint8 as input, otherwise 
    
    if type is None:
       type = 'uint8'
    typedArray = volumeArray.astype(type)
    
    if order is None:
        order = 3
    
    rotatedArray = verticalAffineVolumeTransform(typedArray, offsetStartCoordinate, offsetEndCoordinate, order)
    return rotatedArray

def verticalAffineVolumeTransform(volumeArray, startCoordinate, endCoordinate, order=None):

    """This with help from Daniele Pelliccia, IDTools.com.au"""
    
    dimensions = [volumeArray.shape[0],volumeArray.shape[1],volumeArray.shape[2]]
    p2 = startCoordinate
    p1 = endCoordinate
    
    # get the voxels positioned between p1 and p2
    voxels_crossed = bresenhamline(list(p1), list(p2))

    # Centre of the line between the points
    voxels_crossedIntList  = [[int(j) for j in i]  for i in voxels_crossed]
    centerInput = findMiddle(voxels_crossedIntList)
    
    # Get the max shape of the output array that fit the rotate input array
    # That is sqrt(3)*max(nx,ny,nz) - diagonal of a cube  
    # Very conservative estimate, in reality can do less
    maxsize = int(np.ceil(np.sqrt(3)*max(tuple(dimensions))))

    # This will make the centre of rotation of the output array to be a scaled version of the centre of the input array
    # It works when the centre of rotation is not at the central coordinate
    c_out = np.array((int(centerInput[0]*maxsize/dimensions[0]), \
                      int(centerInput[1]*maxsize/dimensions[1]), \
                      int(centerInput[2]*maxsize/dimensions[2])))
    
    # Define a rotation matrix
    # initialise P1 and P2 as arrays. Set a unit vector.
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    pV = p2 - p1  # The unit vector decribing the points
    
    # Compute the rotation matrix
    rotationMatrix = get_rotation_matrix(pV)
        
    # Calculate offset for affine_transform
    offset=centerInput-c_out.dot(rotationMatrix)

    if order is None:
     order = 3
    
    # Affine transform the array using the rotation matrix and the offset
    rotatedArray=interpolation.affine_transform(volumeArray,rotationMatrix.T,order=order,offset=offset, \
	output_shape=(maxsize,maxsize,maxsize),cval=0.0)

    return rotatedArray
    
def coordinateFlip(coordinates):
    # flips first and third coordinates around; sometimes useful for handling arrays and raw tree surveyor output.
    flippedCoordinates = [coordinates[2],coordinates[1],coordinates[0]]
    return flippedCoordinates
    
def makeSphereArray(sphereSize, blurSigma=0):
    """Generates a spot of spot_size^3 diameter, and optionally blurs it using gaussian filter of given sigma
    """
    diameter = sphereSize
    radius = diameter / 2
    r2 = np.arange(-radius, radius+1)**2
    dist2 = r2[:, None, None] + r2[:, None] + r2
    
    sphere = dist2 <= radius **2
    
    # convert the float array back to integer array
    sphere = np.asarray(sphere, dtype=int)
    sphere[sphere==1] = 255
   
    sphere = filters.gaussian_filter(sphere,blurSigma, mode = 'constant')

    #Returns a numpy array, and the diameter of the array
    return sphere

def spot_stamp(targetArray, coordinates, spotsize, blur=0):
    """Places a spot from make_spot function at a position at a tuple of coordinates defined in the input within an 8 bit target array of any size"""
    # spotsize is the diameter of the spot you want to stamp
    diameter = spotsize
    
    if len(targetArray.shape) > 3:
        print "spot_stamp error! Can't stamp a spot on an RGB Volume"
        sys.exit()
    
    # grab the coordinates from the list
    x, y, z = coordinates[2], coordinates[1], coordinates[0]
    
    # Two ndarrays, of different size.
    #1. target_array, the array you want the sphere to be "stamped" on.
    #2. the sphere array, containing the sphere.
    
    # sphereArray is a fixed size matrix defined by a variable input diameter (spot_size) and gamma correction. Using defaults here
    sphereArray = makeSphereArray(diameter,blur)

    # generate some relative coordinates to locate the sphere, by removing the radius from each source coordinate
    pos_x = int(x - (diameter / 2))
    pos_y = int(y - (diameter / 2))
    pos_z = int(z - (diameter / 2))


    # generate slice objects that don't step over the boundaries of the target array (hopefully!)
    x_range1 = slice(max(0, pos_x), max(min(pos_x + sphereArray.shape[2], targetArray.shape[2]), 0), 1)
    y_range1 = slice(max(0, pos_y), max(min(pos_y + sphereArray.shape[1], targetArray.shape[1]), 0), 1)
    z_range1 = slice(max(0, pos_z), max(min(pos_z + sphereArray.shape[0], targetArray.shape[0]), 0), 1)
    
    x_range2 = slice(max(0, -pos_x), min(-pos_x + targetArray.shape[2], sphereArray.shape[2]), 1)
    y_range2 = slice(max(0, -pos_y), min(-pos_y + targetArray.shape[1], sphereArray.shape[1]), 1)
    z_range2 = slice(max(0, -pos_z), min(-pos_z + targetArray.shape[0], sphereArray.shape[0]), 1)

    # insert the elements of the sphereArray at the location in the targetArray
    targetArray[z_range1, y_range1, x_range1] = sphereArray[z_range2, y_range2, x_range2]

    return targetArray        
    
def intialiseVolumeArray(volume_size_list_xyz):
    """takes in a LIST of size values in format x y z, and returns a 3d array filled with zeroes"""
    x = volume_size_list_xyz[0]
    y = volume_size_list_xyz[1]
    z = volume_size_list_xyz[2]
    zerosArray = np.zeros((x,y,z),dtype=np.int8)
    return zerosArray

def intialiseVolumeArray64(volume_size_list_xyz):
    """takes in a LIST of size values in format x y z, and returns a 3d array filled with zeroes"""
    x = volume_size_list_xyz[0]
    y = volume_size_list_xyz[1]
    z = volume_size_list_xyz[2]
    zerosArray = np.zeros((x,y,z),dtype=np.float64)
    return zerosArray    
    
def bresenhamline(startCoordinates, endCoordinates):
    """takes start and end coordiantes as a LIST in format (x,y,z), appends a dimension (required for the bresenham line code) to the list, and converts the start and end LISTs to a numpy ARRAY (required for the bresenham line code)
    based on:
    http://code.activestate.com/recipes/578112-bresenhams-line-algorithm-in-n-dimensions/
    """
    dimension = 0
    startCoordinates.append(dimension)
    endCoordinates.append(dimension)
    startPoint = np.array([startCoordinates])
    endPoint = np.array([endCoordinates])

    max_iter = np.amax(np.amax(np.abs(endPoint - startPoint), axis=1))
    npts, dim = startPoint.shape

    scale = np.amax(np.abs(endPoint - startPoint), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    nslope = np.array((endPoint - startPoint), dtype=np.double) / scale
    nslope[zeroslope] = np.zeros((endPoint - startPoint)[0].shape)
    
    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = startPoint[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat
    
    # Approximate to nearest int
    # Return the points as a single array
    bresenham_ndCoordinateArray  = np.array(np.rint(bline), dtype=startPoint.dtype).reshape(-1, startPoint.shape[-1])
    
    # turn the array into a list of form (X,Y,Z,axis), and trim the axis off the list to give a list of (X,Y,Z) point coordinates
    coordinategroupsList = np.ndarray.tolist(bresenham_ndCoordinateArray)
    threeCoordinateList = []
    for coordinategroup in coordinategroupsList:
        threeCoordinates = coordinategroup[:-1]
        threeCoordinateList.append(threeCoordinates)
    
    #turn any floats to ints
    threeCoordinateList = [[int(j) for j in i]  for i in threeCoordinateList]
    
    # Return the points as a list of points, representing the traversed voxels between startCoordinates and endCoordinates
    return threeCoordinateList

def branchControlPointSphereMask(nxTreeSurveyorGraph,branchIDlist):
    branchSizeBoost = 0
    pixelsize = 3.21
    dimensions = getVolumeDimensions(nxTreeSurveyorGraph)
    
    generated_array = intialiseVolumeArray64(dimensions)
    zeroArray = intialiseVolumeArray64(dimensions)
    
    if isinstance(branchIDlist, list):
        for idx, branch in enumerate(branchIDlist):
            #print str(idx+1)+" of "+str(len(branchIDlist))
                    
            diameter = int(round(getBranchDiameter(nxTreeSurveyorGraph,branch)/pixelsize))+branchSizeBoost
            splines = getBranchSplineControlPoints(nxTreeSurveyorGraph,branch)
            
            for firstPoint, secondPoint in zip(splines[:-1], splines[1:]):
                coordinateStart = coordinateFlip(firstPoint)
                coordinateEnd = coordinateFlip(secondPoint)
                
                voxels_crossed = bresenhamline(coordinateStart, coordinateEnd)
        
                for point in voxels_crossed:
                    stamped = spot_stamp(np.copy(zeroArray), point, diameter, blur=1)
                    generated_array = np.maximum(stamped, generated_array)
                    stamped = intialiseVolumeArray64(dimensions)

    if isinstance(branchIDlist, str):
        branch = branchIDlist
        diameter = int(round(getBranchDiameter(nxTreeSurveyorGraph,branch)/pixelsize))+branchSizeBoost
        splines = getBranchSplineControlPoints(nxTreeSurveyorGraph,branch)
        
        for firstPoint, secondPoint in zip(splines[:-1], splines[1:]):
            coordinateStart = list(firstPoint)
            coordinateEnd = list(secondPoint)
               
            voxels_crossed = bresenhamline(coordinateStart, coordinateEnd)
            
            for point in voxels_crossed:
                stamped = spot_stamp(np.copy(zeroArray), point, diameter, blur=1)
                generated_array = np.maximum(stamped, generated_array)
                stamped = intialiseVolumeArray64(dimensions)
    
    generated_array.astype(np.int8)
    return generated_array

def branchSplineSphereMask(dimensionInfoArray, nxTreeSurveyorGraph,branchID,sizeBoost=2, cropped=False):
    """
    Given a NetworkX treesurveyor graph, and a single branch ID as a string; this function will generate a numpy 3d dimensionInfoArray with spheres stamped on points of a b-spline curve under the control of the spline control points from the graph edge for the branchID provided.
    Returns the masked sphere array.
    """
    branchSizeBoost = sizeBoost
    pixelsize = 3.21
    dimensions = [dimensionInfoArray.shape[0],dimensionInfoArray.shape[1],dimensionInfoArray.shape[2]]
    # = getVolumeDimensions(nxTreeSurveyorGraph)
    diameter = int(round(getBranchDiameter(nxTreeSurveyorGraph,branchID)/pixelsize))+branchSizeBoost
    
    if cropped==True:
        splines = getBranchSplineControlPoints(nxTreeSurveyorGraph,branchID,cropped=True)
    else:
        splines = getBranchSplineControlPoints(nxTreeSurveyorGraph,branchID)
        
    splineCoordinateArray = bspline(np.asarray(splines))
    splineCoordinateList = np.ndarray.tolist(splineCoordinateArray)
    splineCoordinateIntList  = [[int(j) for j in i]  for i in splineCoordinateList]
    
    generated_array = intialiseVolumeArray64(dimensions)
    zeroArray = intialiseVolumeArray64(dimensions)

    
    for point in splineCoordinateIntList:
        stamped = spot_stamp(np.copy(zeroArray), point, diameter, blur=1)
        generated_array = np.maximum(stamped, generated_array)
        stamped = intialiseVolumeArray64(dimensions)
    
    generated_array.astype(np.int8)

    
    return generated_array
        
def bspline(cv, n=100, degree=5):
    """ Calculate n samples on a bspline

        cv :      Array of control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        https://stackoverflow.com/questions/34803197/fast-b-spline-algorithm-with-numpy-scipy
    """
    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    degree = np.clip(degree,1,count-1)

    # Calculate knot vector
    kv = None
    kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

    # Calculate query range
    u = np.linspace(False,(count-degree),n)

    # Calculate result
    return np.array(si.splev(u, (kv,cv.T,degree))).T

    
# def findAffineTransformedPoints(nxTreeSurveyorGraph, TreeSurveyorBranchID, ndArray):
    
    # """Use to determine the start and end coordinates of a rotated branch after affine transformation
    # This method is kind of sloppy. But it essentially performs the exact same affine transformation as for the whole
    # tip, but in this case, on an array the same size as the whole tip, but with just two pixels marked on it.
    # One, the start, is given a value of 5000.
    # The second, the end point, is given a value of 1000.
    # The affine transformation is done --- this is critical --- WITH AN ORDER OF 0.
    # After afifne transformation, np.argwhere is used to find the new position of the pixels with values 5000 and 1000.
    # Sometimes the process of affine tranformation can create more than one pixel at destination. So we only take the first coordinates returned; and return the location with the SAME X and Z coordinates. Because the rotateVertical function rotates
    # to a unitvector along the Y axis. So the only variation should be in Y after transformation.
    # """
    
    # dimensions = (ndArray.shape[0],ndArray.shape[1],ndArray.shape[2])
    # zerosArray = np.zeros([dimensions[0],dimensions[1],dimensions[2]],dtype=np.float32)
    # startPoint, endPoint = findCropPointOffset(nxTreeSurveyorGraph,TreeSurveyorBranchID)
    # print zerosArray.shape
    # print "start point: "+str(startPoint) + "    end Point: " + str(endPoint)
    
    # zerosArray[startPoint[0],startPoint[1],startPoint[2]] = 5000
    # zerosArray[endPoint[0],endPoint[1],endPoint[2]] = 1000

    # export_array_32bit_byte_raw(zerosArray, "markedZeros.raw")
    
    # rotatedPointMarkers = rotateVertical(np.copy(zerosArray), nxTreeSurveyorGraph, TreeSurveyorBranchID, type=np.float32, order=0)
    # print rotatedPointMarkers.shape
    # export_array_32bit_byte_raw(rotatedPointMarkers, "rotatedPointMarkers.raw")
    
    # startPointLocation = np.argwhere(rotatedPointMarkers == 5000)[0]
    # endPointLocation = np.argwhere(rotatedPointMarkers == 1000)[0]
    # print startPointLocation
    # print endPointLocation
    # #endPointLocation = np.argwhere(rotatedPointMarkers == 1000)[0]

    # #X and Z should be the same between the coordinates due to the rotation to Y unit vector. So remapped coordinates
    # #have the same X and Z, with variable Y.
    
    # remappedStartLocation = (startPointLocation[0],startPointLocation[1],startPointLocation[2])
    # remappedEndLocation = (startPointLocation[0],endPointLocation[1],startPointLocation[2])
    
    # return remappedStartLocation, remappedEndLocation
    

def calculateTransformedCoordinates(nxTreeSurveyorGraph, TreeSurveyorBranchID, ndArray):
    
    """Use to determine the start and end coordinates of a rotated branch after affine transformation
    requires the ndArray of the whole or cropped volume of the array that has ALREADY BEEN TRANSFORMED.
    The coordinates will be centered to the array that is provided.
    
    """
    
    dimensions = (ndArray.shape[0],ndArray.shape[1],ndArray.shape[2])
    offsetStartCoordinate, offsetEndCoordinate = findCropPointOffset(nxTreeSurveyorGraph, TreeSurveyorBranchID)
    p1 = offsetStartCoordinate
    p2 = offsetEndCoordinate
    
    # get the voxels positioned between p1 and p2
    voxels_crossed = bresenhamline(list(p1), list(p2))

    # Centre of the line between the points
    voxels_crossedIntList  = [[int(j) for j in i]  for i in voxels_crossed]
    centerOffsetCoordinate = findMiddle(voxels_crossedIntList)
    
    p1Array = np.asarray(p1)
    p2Array = np.asarray(p2)
    pointsVector = p2Array - p1Array
    rotation_matrix = get_rotation_matrix(pointsVector, unitVector=None)
    
    #Calculate point rotation
    newP1Location = np.dot(p1Array, rotation_matrix.T)
    newP2Location = np.dot(p2Array, rotation_matrix.T)
    
    #Calculate the centre of the rotation transformed points.
    rotatedVoxelsCrossed = bresenhamline(newP1Location.tolist(), newP2Location.tolist())
    rotatedVoxelsCrossedIntList  = [[int(j) for j in i]  for i in rotatedVoxelsCrossed]
    rotatedCenterCoordinate = np.asarray(findMiddle(rotatedVoxelsCrossedIntList))
    
    #calculate translation to move the rotatedCenterCoordinate to the center of the array
    arrayCenter = np.asarray(centreOfArray(ndArray))
    translationCoordinates = rotatedCenterCoordinate - arrayCenter

    #calculate the shifted position of the new coordinates -- not sure which order to do this.
    newTranslatedP1Location = newP1Location - translationCoordinates
    newTranslatedP2Location = newP2Location - translationCoordinates
    
    #the output fromt the previous calculations are floats, so convert to ints
    newP1 = newTranslatedP1Location.astype(int)
    newP2 = newTranslatedP2Location.astype(int)
    
    return newP1, newP2
    
    
    
    
    
    
def centreOfArray(array):
    centerList = []
    for d in array.shape:
        centerOfAxis = np.int(d/2.)
        centerList.append(centerOfAxis)
    return tuple(centerList)
   
    
    
    
    
    
    
    
    
    
    
    
    
##############################################################################################    
    
"""A BUNCH OF FUNCTIONS FOR MAIPULATING COLOURS IN NUMPY ARRAYS"""

def rgbToHSV(rgb_list):
    # Takes a 3 item list of 8 bit values (0-255), in order Red, Green, Blue -- and returns HSV)
    # Created by Victor Lin 
    # from http://code.activestate.com/recipes/576554-covert-color-space-from-hsv-to-rgb-and-rgb-to-hsv/
    # Convert RGB color space to HSV color space
    # using range 0-255
    r, g, b = rgb_list
    
    r = float(r)/255
    g = float(g)/255
    b = float(b)/255
    
    maxc = max(r, g, b)
    minc = min(r, g, b)
    colorMap = {
        id(r): 'r',
        id(g): 'g',
        id(b): 'b'
    }
    if colorMap[id(maxc)] == colorMap[id(minc)]:
        h = 0
    elif colorMap[id(maxc)] == 'r':
        h = ((g - b) * 60.0 / (maxc - minc)) % 360
    elif colorMap[id(maxc)] == 'g':
        h = ((b - r) * 60.0 / (maxc - minc)) + 120
    elif colorMap[id(maxc)] == 'b':
        h = ((r - g) * 60.0 / (maxc - minc)) + 240
    v = maxc
    if maxc == 0.0:
        s = 0.0
    else:
        s = 1 - (minc * 1.0 / maxc)
    h = h/360 * 255
    s = s * 255
    v = v *255
    return (h, s, v)

def rgbArray_2_hsvArray(rgb_array):
    # converts RGB values at each position in a 3d ndArray into HSV values (hue, saturation, and value)
    #dimensions = getVolumeDimensions(nxTreeSurveyorGraph)

    dimensions = [rgb_array.shape[0],rgb_array.shape[1],rgb_array.shape[2]]
        
    hsvArray = np.empty_like(rgb_array)
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            for z in range(dimensions[2]):
                colours = rgb_array[x, y, z]
                if (colours[0] == 0 and colours[1] == 0 and colours[2] == 0):
                    hsvArray[x,y,z][0] = 0
                    hsvArray[x,y,z][1] = 0
                    hsvArray[x,y,z][2] = 0
                else:
                    h, s, v = rgbToHSV(colours)
                    hsvArray[x,y,z][0] = h
                    hsvArray[x,y,z][1] = s
                    hsvArray[x,y,z][2] = v
    return(hsvArray)  

def removeYellowVoxelTree(rgb3DYellowSkeletonArray):
    """ An colour specific RGB median filter for Tree Surveyor RGB data.
    
    When Tree Surveyor exports its RGB volumes, it overlays the skeleton with a yellow voxel tree.
    This is a problem from time to time when trying to do work on the volume
    This function is written to remove the yellow voxel skeleton from volume by applying an RGB median filter
    It trawls iteratively through all dimensions, finds voxels whose values match the yellow colour, and then samples all the values around that voxel, replacing the yellow voxels with the mean RGB values of the neighbours (that aren't black, or yellow)
    
    The RGB ndarray has 4 dimensions, representing an X Y Z, and a C dimension containinig 8bit values for the 3 colours.
    """
 
    #These are the hard-wired yellow colours in Tree Surveyor RGB volumes
    #Red channel = 239
    #Green channel = 239
    #Blue channel = 099
    
    red = 239
    green = 239
    blue = 99
    
    Xdim = rgb3DYellowSkeletonArray.shape[0]
    Ydim = rgb3DYellowSkeletonArray.shape[1]
    Zdim = rgb3DYellowSkeletonArray.shape[2]
    
    for x in range(Xdim):
        for y in range(Ydim):
            for z in range(Zdim):
                colours = rgb3DYellowSkeletonArray[x, y, z]
                # find the yellow skeleton at this position
                if (colours[0] == red and colours[1] == green and colours[2] == blue):
                    # grab the colours of the immediate pixels surrounding the yellow skeleton pixel stamped by Tree Surveyor
                    # notation means Z = Z position. M = middle. T = top. L = left. R = Right. In relation to the yellow pixel.
                    
                    if x == 0:
                        xback = 0
                    else:
                        xback = x-1
                    
                    if y == 0:
                        yback = 0
                    else:
                        yback = y-1
                    
                    if z == 0:
                        zback = 0
                    else:
                        zback = z-1
                                        
                    if x >= Xdim - 1:
                        xforward = Xdim-1
                    else:
                        xforward = x+1
                    
                    if y >= Ydim-1:
                        yforward = Ydim-1
                    else:
                        yforward = y+1
                    
                    if z >= Zdim - 1:
                        zforward = Zdim-1
                    else:
                        zforward = z+1
                        
                    coloursZMTL = rgb3DYellowSkeletonArray[xback, yback, z]
                    coloursZMTM = rgb3DYellowSkeletonArray[x, yback, z]
                    coloursZMTR = rgb3DYellowSkeletonArray[xforward, yback, z]
                    coloursZMML = rgb3DYellowSkeletonArray[xback, y, z]
                    coloursZMMR = rgb3DYellowSkeletonArray[xforward, y, z]
                    coloursZMBL = rgb3DYellowSkeletonArray[xback, yforward, z]
                    coloursZMBM = rgb3DYellowSkeletonArray[x, yforward, z]
                    coloursZMBR = rgb3DYellowSkeletonArray[xforward, yforward, z]
                    
                    coloursZTTL = rgb3DYellowSkeletonArray[xback, yback, zforward]
                    coloursZTTM = rgb3DYellowSkeletonArray[x, yback, zforward]
                    coloursZTTR = rgb3DYellowSkeletonArray[xforward, yback, zforward]
                    coloursZTML = rgb3DYellowSkeletonArray[xback, y, zforward]
                    coloursZTMM = rgb3DYellowSkeletonArray[x, y, zforward]
                    coloursZTMR = rgb3DYellowSkeletonArray[xforward, y, zforward]
                    coloursZTBL = rgb3DYellowSkeletonArray[xback, yforward, zforward]
                    coloursZTBM = rgb3DYellowSkeletonArray[x, yforward, zforward]
                    coloursZTBR = rgb3DYellowSkeletonArray[xforward, yforward, zforward]
                    
                    coloursZBTL = rgb3DYellowSkeletonArray[xback, yback, zback]
                    coloursZBTM = rgb3DYellowSkeletonArray[x, yback, zback]
                    coloursZBTR = rgb3DYellowSkeletonArray[xforward, yback, zback]
                    coloursZBML = rgb3DYellowSkeletonArray[xback, y, zback]
                    coloursZBMM = rgb3DYellowSkeletonArray[x, y, zback]
                    coloursZBMR = rgb3DYellowSkeletonArray[xforward, y, zback]
                    coloursZBBL = rgb3DYellowSkeletonArray[xback, yforward, zback]
                    coloursZBBM = rgb3DYellowSkeletonArray[x, yforward, zback]
                    coloursZBBR = rgb3DYellowSkeletonArray[xforward, yforward, zback]

                    # gather the red channel intensity values surrounding the yellow pixel
                    medianRedValues = np.array([coloursZMTL[0], coloursZMTM[0], coloursZMTR[0], coloursZMML[0], coloursZMMR[0], coloursZMBL[0], coloursZMBM[0], coloursZMBR[0], coloursZTTL[0], coloursZTTM[0], coloursZTTR[0], coloursZTML[0], coloursZTMM[0], coloursZTMR[0], coloursZTBL[0], coloursZTBM[0], coloursZTBR[0], coloursZBTL[0], coloursZBTM[0], coloursZBTR[0], coloursZBML[0], coloursZBMM[0], coloursZBMR[0], coloursZBBL[0], coloursZBBM[0], coloursZBBR[0]])

                    # gather the green channel intensity values surrounding the yellow pixel
                    medianGreenValues = np.array([coloursZMTL[1], coloursZMTM[1], coloursZMTR[1], coloursZMML[1], coloursZMMR[1], coloursZMBL[1], coloursZMBM[1], coloursZMBR[1], coloursZTTL[1], coloursZTTM[1], coloursZTTR[1], coloursZTML[1], coloursZTMM[1], coloursZTMR[1], coloursZTBL[1], coloursZTBM[1], coloursZTBR[1], coloursZBTL[1], coloursZBTM[1], coloursZBTR[1], coloursZBML[1], coloursZBMM[1], coloursZBMR[1], coloursZBBL[1], coloursZBBM[1], coloursZBBR[1]])

                    # gather the blue channel intensity values surrounding the yellow pixel
                    medianBlueValues = np.array([coloursZMTL[2], coloursZMTM[2], coloursZMTR[2], coloursZMML[2], coloursZMMR[2], coloursZMBL[2], coloursZMBM[2], coloursZMBR[2], coloursZTTL[2], coloursZTTM[2], coloursZTTR[2], coloursZTML[2], coloursZTMM[2], coloursZTMR[2], coloursZTBL[2], coloursZTBM[2], coloursZTBR[2], coloursZBTL[2], coloursZBTM[2], coloursZBTR[2], coloursZBML[2], coloursZBMM[2], coloursZBMR[2], coloursZBBL[2], coloursZBBM[2], coloursZBBR[2]])
                  
                    #test to make sure any of the sampled colours aren't yellow pixels, if they are, delete them from the array
                    yellow_index1 = (medianRedValues == red).nonzero()
                    yellow_index2 = (medianGreenValues == green).nonzero()
                    yellow_index3 = (medianBlueValues == blue).nonzero()
                    filteredReds = np.delete(medianRedValues, yellow_index1)
                    filteredGreens = np.delete(medianGreenValues, yellow_index2)
                    filteredBlues = np.delete(medianBlueValues, yellow_index3)

                    # if the array is empty, you can't use numpy mean, so check to make sure it's non zero first.
                    if filteredReds.size != 0:
                        meanRed = int(np.mean(filteredReds))
                    if filteredGreens.size != 0:
                        meanGreen = int(np.mean(filteredGreens))
                    if filteredBlues.size != 0:
                        meanBlue = int(np.mean(filteredBlues))

                    # apply the mean colours to the yellow pixel
                    rgb3DYellowSkeletonArray[x,y,z][0] = meanRed
                    rgb3DYellowSkeletonArray[x,y,z][1] = meanGreen
                    rgb3DYellowSkeletonArray[x,y,z][2] = meanBlue
    
    # at the end of the loop, all the yellow voxels have been removed with an RGB median filtering
    rgbNoYellowArray = rgb3DYellowSkeletonArray
    
    # the return name is unfortunate, because this returns an array WITHOUT the yellow skeleton
    return rgbNoYellowArray

    



















    
#############################################################################################################################    
"""A bunch of funcions for segmenting, adding, masking and cropping arrays"""



def segmentArray(array, colourSampleCoordinates):
    """This is used as a part of the treesurveyor RGB to segmented branch code."""

    #initialise an emptpy array of zeroes that are the same dimensions as the source volume
    dimensions = [array.shape[0],array.shape[1],array.shape[2]]
    
    eightbitArray = np.zeros((dimensions[0],dimensions[1],dimensions[2]), dtype=np.int)

    #identify the HSV value at the colourSampleCoordinates provided
    hsvAtPoint = array[colourSampleCoordinates[0],colourSampleCoordinates[1],colourSampleCoordinates[2]]
    
    hue = hsvAtPoint[0]
    saturation = hsvAtPoint[1]
    intensity_value = hsvAtPoint[2]
    
    # this is how much variation to allow the HSV matching to include. 
    variance = 42
    
    saturationHi = saturation + variance
    saturationLow = max(saturation - variance,0)
    intensity_valueHi = intensity_value + variance
    intensity_valueLow = max(intensity_value - variance,0)
    
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            for z in range(dimensions[2]):
                if array[x,y,z][0] == hue and saturationLow <= array[x,y,z][1] <= saturationHi and intensity_valueLow <= array[x,y,z][2] <= intensity_valueHi:
                    
                    eightbitArray[x,y,z] = 255
    
    # exports array that is iso surfaced, whereever the colour matches.
    return eightbitArray

def autoCrop3d(ndArray):
    boundingBoxCoordinates = boundingBox3D(ndArray)
    croppedNdArray = cropArray(ndArray, boundingBoxCoordinates)
    return croppedNdArray
    
def boundingBox3D(ndArray):
    #https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    
    # axis 0 = x; 1 = y, 2 = z
    yz = np.any(ndArray, axis=(1, 2))
    xz = np.any(ndArray, axis=(0, 2))
    xy = np.any(ndArray, axis=(0, 1))

    xmin, xmax = np.where(yz)[0][[0, -1]]
    ymin, ymax = np.where(xz)[0][[0, -1]]
    zmin, zmax = np.where(xy)[0][[0, -1]]

    points = [int(xmin), int(xmax), int(ymin), int(ymax), int(zmin), int(zmax)]
    return points
    
def cropArray(ndArray, points):
    """takes an ndArray three dimensional array, and a list of points and crops them. Need min and max for each dimension"""
    # Points should be in order (Xmin, Xmax, Ymin, Ymax, Zmin, Zmax)
    
    Xmin = int(points[0])
    Xmax = int(points[1])
    Ymin = int(points[2])
    Ymax = int(points[3])
    Zmin = int(points[4])
    Zmax = int(points[5])
    
    if len(ndArray.shape) == 3:
        cropped_ndArray = ndArray[Xmin:Xmax, Ymin:Ymax, Zmin:Zmax]
    if len(ndArray.shape) == 4:
        cropped_ndArray = ndArray[Xmin:Xmax, Ymin:Ymax, Zmin:Zmax,]
    return cropped_ndArray

def cropPoints(nxTreeSurveyorGraph, TreeSurveyorBranchID):
    """This crops a single segment from a Tree Surveyor ndarray
    It grabs the spline control points to provide the information for the extreme x,y,z points of the branch.
    It also adds on an extra margin of 10 voxels around it.
    
    To extract a clean unencumbered segment, it is best to feed it a previously masked segment free of contaminating
    branch/tips.
    """
        
    BranchSplinePoints = getBranchSplineControlPoints(nxTreeSurveyorGraph,TreeSurveyorBranchID)
    
    margin = 10 #voxels
    
    # identify the spline points
    splineCoordinateArray = bspline(np.asarray(BranchSplinePoints))
    # the result is an array of nested points. Just flatten it.
    flatCoords = np.ravel(splineCoordinateArray)
    
    # grab every 1st (X), 2nd(Y) and 3rd(Z) points, iterating every 3 points through the whole array of points.
    X_coords = flatCoords[0::3]
    Y_coords = flatCoords[1::3]
    Z_coords = flatCoords[2::3]
    
    # Identify the largest and smallest bounds of the spline, expand the volume by the margin.
    Xmax = int(X_coords.max()) + margin
    Ymax = int(Y_coords.max()) + margin
    Zmax = int(Z_coords.max()) + margin
    Xmin = int(X_coords.min()) - margin
    Ymin = int(Y_coords.min()) - margin
    Zmin = int(Z_coords.min()) - margin
    
    # Test to make sure the Max and Min values don't lie outside the volume they are to be cropped from.
    # by default, use the whole input volume dimensions.
    dimensions = getVolumeDimensions(nxTreeSurveyorGraph)
    
    Xdim = dimensions[0]
    Ydim = dimensions[1]
    Zdim = dimensions[2]
    
    if Xmin < 0:
        Xmin = 0
    if Xmax > Xdim:
        Xmax = Xdim

    if Ymin < 0:
        Ymin = 0
    if Ymax > Ydim:
        Ymax = Ydim

    if Zmin < 0:
        Zmin = 0
    if Zmax > Zdim:
        Zmax = Zdim
    
    points = [Xmin,Xmax,Ymin,Ymax,Zmin,Zmax]

    return points

def findCropOffset(nxTreeSurveyorGraph, TreeSurveyorBranchID):
    croppingPoints = cropPoints(nxTreeSurveyorGraph, TreeSurveyorBranchID)
    cropOffset = [croppingPoints[0],croppingPoints[2],croppingPoints[4]]
    return cropOffset
    
def findCropPointOffset(nxTreeSurveyorGraph, TreeSurveyorBranchID):
    branchVolumeCropOffset = findCropOffset(nxTreeSurveyorGraph,TreeSurveyorBranchID)
    startCoordinate = getBaseCoordinate(nxTreeSurveyorGraph, TreeSurveyorBranchID)
    endCoordinate = getEndCoordinate(nxTreeSurveyorGraph, TreeSurveyorBranchID)
    
    offsetStartCoordinate = [int(startCoordinate[0]-int(branchVolumeCropOffset[0])),int(startCoordinate[1]-int(branchVolumeCropOffset[1])),int(startCoordinate[2]-int(branchVolumeCropOffset[2]))]
    offsetEndCoordinate = [int(endCoordinate[0]-int(branchVolumeCropOffset[0])),int(endCoordinate[1]-int(branchVolumeCropOffset[1])),int(endCoordinate[2]-int(branchVolumeCropOffset[2]))]
    
    #return the start and then relocated point endCoordinates 
    return offsetStartCoordinate, offsetEndCoordinate

def cropSegment(nxTreeSurveyorGraph, TreeSurveyorBranchID, ndArray):
    points = cropPoints(nxTreeSurveyorGraph, TreeSurveyorBranchID)
    cropped_ndArray = cropArray(ndArray, points)
    
    # if you're going to load the RAW array into Fiji/ImageJ, you need to know the dimensions of the cropped array, so print them here.
    return cropped_ndArray

def ArrayMaskWhite(ndarray1, ndarray2):
    ''' Compares two EQUAL DIMENSION arrays. If non-zero values are identified at matching coordinates/indexes, a newly created array will be generated with a mask representing the overlapping regions of both
returns an ndArray of same shape, filled with 0s and 255s depending on matches'''

    if not ndarray1.shape == ndarray2.shape:
        print "ArrayMaskWhite function: input arrays are not the same shape"
        print "First array shape: "+ str(ndarray1.shape)+"   Second array shape: " + str(ndarray2.shape)
        sys.exit()
    
    ndArrayAND = intialiseVolumeArray([ndarray1.shape[0],ndarray1.shape[1],ndarray1.shape[2]])

    for index, x in np.ndenumerate(ndarray1):
        if int(x) & int(ndarray2[index]):
            ndArrayAND[index] = 255

    return ndArrayAND


def ArrayMaskGrey(ndarray1, ndarray2):
    ''' Compares two EQUAL DIMENSION arrays. If non-zero values are identified at matching coordinates/indexes, a newly created array will be generated with a mask representing the overlapping regions of both
returns an ndArray of same shape, filled with 0s and 255s depending on matches'''
    
    if not ndarray1.shape == ndarray2.shape:
        print "ArrayMaskGrey function: input arrays are not the same shape"
        print "First array shape: "+ str(ndarray1.shape)+"   Second array shape: " + str(ndarray2.shape)
        sys.exit()
    ndArrayAND = intialiseVolumeArray([ndarray1.shape[0],ndarray1.shape[1],ndarray1.shape[2]])
    for index, x in np.ndenumerate(ndarray1):
        if int(x) & int(ndarray2[index]):
            ndArrayAND[index] = ndarray2[index]
    
    return8bitUnsigned = ndArrayAND.astype('uint8')
    return return8bitUnsigned
 
    
    
    
    
    
    
    
    
    
    
    
################################################################################################################
"""A BUNCH OF IO FUNCTIONS FOR HANDLING TREE SURVEYOR INPUT DATA AND EXPORTING RAW DATA"""

def loadGraphML(GraphML_file):
    # attempt to load a Tree Surveyor graphml file. Sometimes if there is not a bifurcation, angle keys (which are set as Double in the Tree Surveyor GraphML key description) become contaminated with "no bifurcation" strings, there are a few instances of "#IND" appearing too.
    # This routine attempts to load as-is, and if it fails, the routine loads the graphml as a string into memory using StringIO, removing all "no bifurcation" and "#IND" lines in the process
    # The altered graphml is then loaded into nxGraph, the altered graphml is removed from memory, and the compliant graphML is exported.
    
    try:
        nxGraphFile = nx.read_graphml(GraphML_file, str)
    except:
        # this will happen usually if some of the keys contain non-numerical values, which are sometimes exported in node keys by Tree Surveyor.
        output_graph_buffer = StringIO.StringIO()
        with open(GraphML_file) as inputGraphML:
            for line in inputGraphML:
                if not 'no bifurcation' in line:
                    if not '#IND' in line:
                        output_graph_buffer.write(line)
        output_graph_buffer.seek(0)
        nxGraphFile = nx.read_graphml(output_graph_buffer, str)
        output_graph_buffer.close()
    return(nxGraphFile)
    
def export_array_8bit_byte_raw(numpy_array,filename):
    """Write a numpy array into a file, of type unsigned 8bit integer."""
    rotated_array = np.swapaxes(numpy_array,0,2)
    eightBitArray = rotated_array.astype('uint8')
    eightBitArray.tofile(filename)

    return
    
def export_array_32bit_byte_raw(numpy_array,filename):
    """Write a numpy array into a file, of type unsigned 8bit integer."""
    rotated_array = np.swapaxes(numpy_array,0,2)
    eightBitArray = rotated_array.astype(np.uint32)
    eightBitArray.tofile(filename)

    return

def loadRaw8bitVolume(rawVolumeFile,nxTreeSurveyorGraph):
    raw = np.fromfile(rawVolumeFile, dtype=np.uint8)
    raw = np.delete(raw, [0,1,2,3,4,5,6,7,8,9,10,11], axis=0)
    
    dimensions = getVolumeDimensions(nxTreeSurveyorGraph)
    rawArray3d = raw.reshape(dimensions[2], dimensions[1], dimensions[0])
    rawArray3d_rotated = np.swapaxes(rawArray3d,0,2)
    
    return rawArray3d_rotated
    
def loadRGBraw(rawRGBfile, fileType):
    """This loads an RGB raw file, but since the two formats are a little different, you need to specify which type """
    if fileType == 'TS':
        #print 'The volume is being treated as a', fileType,'RGB volume'
        red_1D_channel, green_1D_channel, blue_1D_channel = raw_32bitRGB_loader(rawRGBfile)
        return(red_1D_channel, green_1D_channel, blue_1D_channel) 
    else:
        #print 'The volume is being treated as an', fileType,'RGB volume'
        red_1D_channel, green_1D_channel, blue_1D_channel = raw_24bitRGB_loader(rawRGBfile)
        return(red_1D_channel, green_1D_channel, blue_1D_channel)
        
def raw_24bitRGB_loader(rawfile):
    """ Takes 24bit raw RGB file and splits the interleaved channels, exporting them as independent red, green and blue arrays."""
    raw = np.fromfile(rawfile, dtype=np.uint8)
        
    raw_red = raw[0::3]
    raw_green = raw[1::3]
    raw_blue = raw[2::3]
    
    return(raw_red,raw_green,raw_blue)

def raw_32bitRGB_loader(TreeSurveyor_32bit_rawfile):
    """This loads a Tree Surveyor 32bit RGB raw file, which is what Tree Surveyor exports as its RGB format (RGBA)
    It trims the 12 byte header that Tree Surveyor uses, and puts the interleaved bytes into separate red, green and blue arrays, ignoring every fourth 8-bit byte which is the alpha channel"""
    
    # takes 32bit raw file from Tree Surveyor and splits the channels, exporting them as independent red, green and blue arrays. It 
    raw = np.fromfile(TreeSurveyor_32bit_rawfile, dtype=np.uint8)
    # remove the 12 byte file header from the raw array
    
    raw = np.delete(raw, [0,1,2,3,4,5,6,7,8,9,10,11], axis=0)
    
    #The input 32bit RGB raw has 4 channels, in dimensions defined by Xdim, Ydim, Zdim (extracted from the GraphML).
    #Pixels are represented as 4 x 8 bit bytes per pixel.
    #The bytes are in order: RED, GREEN, BLUE and ALPHA.
    #To extract the channels separately, extract every 4th byte from the array starting from position 0, 1, 2, 3
    
    raw_red = raw[0::4]
    raw_green = raw[1::4]
    raw_blue = raw[2::4]
    #raw_alpha = raw[3::4]
    return(raw_red,raw_green,raw_blue)

def convert1dRGBto3D(rawR, rawG, rawB, nxTreeSurveyorGraph):
    # imported RGB data from loadRGBraw() [i.e. raw_24bitRGB_loader() and raw_32bitRGB_loader()] is a 1D array. 
    # This function reshapes the data to a 3D array based on the GraphML input from GraphML Xdim, Ydim, Zdim Grid Size keys
    # The RGB ndarray has 4 dimensions after stacking, representing an X Y Z, and a C dimension containinig 8bit values for the 3 colours.
    
    dimensions = getVolumeDimensions(nxTreeSurveyorGraph)
    
    redArray3d = rawR.reshape(dimensions[2], dimensions[1], dimensions[0])
    greenArray3d = rawG.reshape(dimensions[2], dimensions[1], dimensions[0])
    blueArray3d = rawB.reshape(dimensions[2], dimensions[1], dimensions[0])
    
    RGBarray3d = np.stack((redArray3d, greenArray3d, blueArray3d), axis=-1)
    RGBarray3d_rotated = np.swapaxes(RGBarray3d,0,2)
    
    #RGBarray3d_rotated_flipped = np.flipud(RGBarray3d_rotated)
    
    return RGBarray3d_rotated
    
def loadRGBrawVolume(rawRGBfile, nxTreeSurveyorGraph, RGBrawFileType):
    red_1D_channel, green_1D_channel, blue_1D_channel = loadRGBraw(rawRGBfile, RGBrawFileType)
    rgb3DArray = convert1dRGBto3D(red_1D_channel, green_1D_channel, blue_1D_channel, nxTreeSurveyorGraph)
    return rgb3DArray
    
def purge(dir, pattern):
    """Deletes a bunch of files in a directory, with patter match - eg .raw"""
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


    
    



##################################################################################################    

""" A BUNCH OF GRAPHML MANIPULATION FUNCTIONS USING networkX"""

def addGenerationToGraphML(nxTreeSurveyorGraph):
    for i in nxTreeSurveyorGraph.node:
        generation = len(str(i))-str(i).count('.')
        if str(i).startswith('Extension Of'):
            generation = generation - 13 #(to make the same as its parent, because it is a projection only)
        if not ('unnamed' in str(i) or 'Root' in str(i)):
            nxTreeSurveyorGraph.add_node(str(i),Generation=generation)
            
def saveGraphML(nxTreeSurveyorGraph, graphFilename='exportGraph.graphml'):
    nx.write_graphml(nxTreeSurveyorGraph, graphFilename)
    return
    
    
    

    
    
 
 
 
    
###################################################################################################    
    
""" SOME GRAPH MODELING FUNCTIONS FOR FINDING PERFECT TREES, HALF DELAY TREES, and FIBONNACCI TREES"""

def getSubtreesMatchingModel(nxTreeSurveyorGraph,pattern="HD",weightMin=1,weightMax=1000):
    if pattern=="PERFECT":
        offsets = [1,1]
        wts = [1,2,4]
    elif pattern=="FIB":
        offsets = [1,2]
        wts = [1,1,2]
    elif pattern=="HD":
        offsets = [2,3]
        wts = [1,1,1]
    else:
        print("!!! Unknown pattern !!!")
        return()
    offsets.sort() # just so we can definitely rely on the order later
    nextWt = 0
    while nextWt < weightMax:
        nextWt = wts[-offsets[0]] + wts[-offsets[1]]
        wts.append(nextWt)
    ret = getSubtreesMatchingModel1(nxTreeSurveyorGraph,"Root",offsets,wts,weightMin,weightMax)
    # need to account for the possibility that whole tree matches pattern, but smaller than weightMin
    if (ret[1]<weightMin):
        return([[]])
    return(ret[2])
   
 
def getSubtreesMatchingModel1(nxTreeSurveyorGraph,rootNode,offsets,wts,weightMin,weightMax):
    # returns [fitsPattern, weight, adj]; if !fitsPattern, adj contains any subtrees that fit criteria 
    # print(rootNode)
    children = nxTreeSurveyorGraph.adj[rootNode].keys()
    if (len(children)==0):
        print("!!! Node with no children - shouldn't happen !!!")
        return()
    elif (len(children)==1):
        return([True,1,[]])
    elif (len(children)>2):
        wt = 0
        edg = []
        for ch in children:
            f,w,e = getSubtreesMatchingModel1(nxTreeSurveyorGraph,ch,
                                              offsets,wts,weightMin,weightMax)
            wt += w
            if (w >= weightMin):
                edg += e
        return([False,wt,edg])
    
    # now the main case - two children
    # can't initially filter by weightMin, since smaller trees required for bigger ones
    # instead filter at end, when parent is non-matching or when doing the final return from top level function
    fitsPattern1, weight1, adj1 = getSubtreesMatchingModel1(nxTreeSurveyorGraph,children[0],
                                                          offsets,wts,weightMin,weightMax)
    fitsPattern2, weight2, adj2 = getSubtreesMatchingModel1(nxTreeSurveyorGraph,children[1],
                                                          offsets,wts,weightMin,weightMax)
    weight = weight1 + weight2
    if (weight <= weightMax) & fitsPattern1 & fitsPattern2 & (weight in wts):
        pos = wts.index(weight) # if multiple entries of weight in list, doesn't actually matter which position we get
        if ([wts[pos-offsets[1]],wts[pos-offsets[0]]] == sorted([weight1,weight2])):
            return([True,weight,adj1+adj2+children])  
    # left with case where node doesn't match, so have to filter subtrees on weightMin
    ed = []
    if weight1 >= weightMin:
        ed += adj1
    if weight2 >= weightMin:
        ed += adj2
    return([False,weight,ed])