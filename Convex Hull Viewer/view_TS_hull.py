import networkx as nx
import argparse
import tree_routines_X3 as tR

def main():
    parser = argparse.ArgumentParser(prog='TipSeg', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('graph_file')
    args = parser.parse_args()
    graphML = args.graph_file
    
    nxGraphML = tR.loadGraphML(graphML)
    tR.plotKidneyHull(nxGraphML)
main()
