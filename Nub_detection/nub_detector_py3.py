import networkx as nx
import argparse
import csv
import sys
import io
sys.path.append('../GraphML_Routine_module/')
import tree_routines_X3_py3 as tx

def main():
    '''
    A tool for detecting tips that are internally located. Basically, it checks its neighbour, and sees how many derivatives there are.
    If there is a lot, then chances are it's an internal tip.
    This will export a csv file, with a list of Tree Surveyor BranchIDs. Next to it, a number of derivatives.
    Use this with Tree Surveyor's "Skeleton Curation", enable "Corrected segment data" visualisation. Hit the "HOME" key, and enter in the BranchIDs with lots of derivatives. Then, manually curate the tip.
    '''

    parser = argparse.ArgumentParser(prog='TipSeg', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('graph_file')
    args = parser.parse_args()
    graphML = args.graph_file

    nxGraphML = tx.loadGraphML(graphML)
    
    tipList = tx.getTipIDs(nxGraphML)
    
    with open('nubDataTest.csv', 'w', newline='') as csvfile:
        dataWriter = csv.writer(csvfile, delimiter =',')
        dataWriter.writerow(['BranchID','Neighbour', 'Neighbour Derivatives', 'Derivative IDs'])
        
        for tip in tipList:
            
            incoming, neighbour = tx.getBranchNeighbourID(nxGraphML, tip)
            
            #print(tip, neighbour)
            descendents = tx.find_descendant_edges(nxGraphML, neighbour)
            
            #print(descendents)
            
            descendents = list(descendents)
            #print(descendents)
            num_descendents = len(descendents)
            #print(num_descendents)
            
            if num_descendents > 0:
            #Prepare the data to be written to the CSV file
                branchdata = [tip, neighbour, num_descendents, descendents]
                dataWriter.writerow(branchdata)
            
main()
