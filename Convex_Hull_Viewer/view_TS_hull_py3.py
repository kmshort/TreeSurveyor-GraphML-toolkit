import networkx as nx
import argparse
import sys
sys.path.append('../GraphML_Routine_module/')

import tree_routines_X3_py3 as tR

def main():
    parser = argparse.ArgumentParser(prog='TipSeg', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('graph_file')
    parser.add_argument('res')
    args = parser.parse_args()
    graphML = args.graph_file
    resolution = args.res
    
    nxGraphML = tR.loadGraphML(graphML)
    tR.plotKidneyHull(nxGraphML)
    vol = tR.getKidneyHullVolume(nxGraphML, resolution)
    sa = tR.getKidneyHullSurfaceArea(nxGraphML, resolution)
    print("Convex hull volume is: " + str(vol))
    print("Convex hull surface area is: " + str(sa))
main()
