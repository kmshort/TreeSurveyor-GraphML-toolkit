import networkx as nx
import numpy as np
import argparse
import tree_routines_X3 as tR
import csv

def main():
    parser = argparse.ArgumentParser(prog='TipSeg', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('graph_file')
    args = parser.parse_args()
    graphML = args.graph_file

    
    
    nxGraphML = tR.loadGraphML(graphML)
    
    tipList = tR.getTipIDs(nxGraphML)
    

    
    with open('nubData.csv', 'wb') as csvfile:
        dataWriter = csv.writer(csvfile, delimiter =',')
        dataWriter.writerow(['BranchID','Neighbour Derivatives'])
        for tip in tipList:
            neighbourTip = tR.getBranchNeighbourID(nxGraphML, tip)
            derivativeTips = [i for i, s in enumerate(tipList) if neighbourTip in s]
            
            t = str(len(derivativeTips))
        
            branchdata = [tip, t]
            dataWriter.writerow(branchdata)
            
            # write raw to minc converter line for Mattijs
            # convert um dimensions to millimetre, used by rawtominc in linux
    tR.plotKidneyHull(nxGraphML)
main()
