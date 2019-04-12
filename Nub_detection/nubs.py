import networkx as nx
import argparse
import csv
import tree_routines_X3 as tR


def main():
    '''
    A rough tool for detecting tips that are internally located. Basically, it checks its neighbour, and sees how many derivatives there are.. if it's a lot, then chances are it's an internal tip.
    This will export a csv file, with a list of Tree Surveyor BranchIDs. Next to it, a number of derivatives.
    Use this with Tree Surveyor's "Skeleton Curation", enable "Corrected segment data" visualisation. Hit the "HOME" key, and enter in the BranchIDs with lots of derivatives. Then, manually curate the tip.
    '''

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
            
main()
