from k_path_dp import DP_from_coloring
import math
import subprocess
import sys
from methods import *





def approxTrials(k, delta):
    p = math.sqrt(2*math.pi*k)/(math.e**k)
    trials = 1/p*math.log(1/delta)
    return trials


def evalColoring(edge_file,k, delta, color_method, total = None):
    if total==None:
        total = 1
    T = approxTrials(k, delta)
    t=0
    Found = False
    sol_set = set()
    while Found is not True:
        if ".py" in color_method:
            coloring = method2(edge_file,k)
            #Change below to function import called instead; subprocess string is tricky
            #coloring = subprocess.run(['python3', color_method, edge_file], stderr=subprocess.PIPE,stdout=subprocess.PIPE)
            sol = DP_from_coloring(edge_file, k ,coloring = coloring)
        else:
            sol = DP_from_coloring(edge_file, k ,type = color_method)
        t = t+1
        if sol!=[]:
            for n in sol:
                print(n)
                sol_set.add(tuple(n))
        if len(sol_set)>= total:
            Found = True
    print("Random trials expected to need "+str(T)+" trials")
    print("Current coloring achieves "+str(t)+" trials")
    return sol_set, T,t


def main():
    if len(sys.argv)<3:
    # python eval.py data/test_example_list 3 0.01 random 
    # python eval.py data/test_example_list 3 0.01 method1.py 100 
        print("Usage "+ sys.argv[0]+ " <graph-file.edge> <k> <delta> <opt: coloring-type> <opt: coloring-file> <opt: total paths>")     #example file: "data/test_example_list"
    else:
        if len(sys.argv)==6:
            return evalColoring(sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4],total =sys.argv[5])
        return evalColoring(sys.argv[1], int(sys.argv[2]),float(sys.argv[3]),sys.argv[4])


if __name__=="__main__":
    main()
