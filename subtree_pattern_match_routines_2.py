def loadGraphML(TreeSurveyor_GraphML_file):
    # attempt to load a Tree Surveyor graphml file. Sometimes if there is not a bifurcation, angle keys (which are set as Double in the Tree Surveyor GraphML key description) become contaminated with "no bifurcation" strings, there are a few instances of "#IND" appearing too.
    # This routine attempts to load as-is, and if it fails, the routine loads the graphml as a string into memory using StringIO, removing all "no bifurcation" and "#IND" lines in the process
    # The altered graphml is then loaded into nxGraph, the altered graphml is removed from memory, and the compliant graphML is exported.
    
    try:
        nxTreeSurveyorGraph = nx.read_graphml(TreeSurveyor_GraphML_file, str)
    except:
        # this will happen usually if some of the keys contain non-numerical values, which are sometimes exported in node keys by Tree Surveyor.
        output_graph_buffer = StringIO.StringIO()
        with open(TreeSurveyor_GraphML_file) as inputGraphML:
            for line in inputGraphML:
                if not 'no bifurcation' in line:
                    if not '#IND' in line:
                        output_graph_buffer.write(line)
        output_graph_buffer.seek(0)
        nxTreeSurveyorGraph = nx.read_graphml(output_graph_buffer, str)
        output_graph_buffer.close()
    return(nxTreeSurveyorGraph)

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
    # returns [fitsPattern, weight, edges]; if !fitsPattern, edges contains any subtrees that fit criteria 
    # print(rootNode)
    children = nxTreeSurveyorGraph.edge[rootNode].keys()
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
    fitsPattern1, weight1, edges1 = getSubtreesMatchingModel1(nxTreeSurveyorGraph,children[0],
                                                          offsets,wts,weightMin,weightMax)
    fitsPattern2, weight2, edges2 = getSubtreesMatchingModel1(nxTreeSurveyorGraph,children[1],
                                                          offsets,wts,weightMin,weightMax)
    weight = weight1 + weight2
    if (weight <= weightMax) & fitsPattern1 & fitsPattern2 & (weight in wts):
        pos = wts.index(weight) # if multiple entries of weight in list, doesn't actually matter which position we get
        if ([wts[pos-offsets[1]],wts[pos-offsets[0]]] == sorted([weight1,weight2])):
            return([True,weight,edges1+edges2+children])  
    # left with case where node doesn't match, so have to filter subtrees on weightMin
    ed = []
    if weight1 >= weightMin:
        ed += edges1
    if weight2 >= weightMin:
        ed += edges2
    return([False,weight,ed])
