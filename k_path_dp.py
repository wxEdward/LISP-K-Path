import random
import sys
from pprint import pprint

def DP_from_coloring(file, k ,type=None,coloring = None):
    fin = open(file)
    #n, m=map(int, next(fin).strip("\n").split())    #first row: #of nodes    #of edges
    n, m=map(int, next(fin).split())
    # print(n,m)
    G={}
    for i in range(m):
        u,v = map(int,next(fin).split())
        if not u in G:
            G[u]=set([])
        G[u].add(v)
    V = list(range(n))
    for u in V:
        if not u in G:
            G[u]=set([])                   #changed index from i to u to add empty set for nodes that don't have outgoing edge

    k = int(k)
    if not coloring:
        if type=="random":
            coloring = {u:random.randint(1,k) for u in V}    #random coloring
        elif type=="method_1":
            coloring = {u:random.randint(1,k) for u in V}    #placeholder for coloring methods (maybe the coloring can be saved and be read here?)

    #coloring = {0: 3, 1: 1, 2: 2, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 1} #hard coded an example to test that correct k-paths are found: [3, 5, 7], [5, 7, 8]
    dp_table = {}
    for u in V:
        dp_table[(u,0)]=set([(coloring[u],)])

    partial_paths = {(u,(coloring[u],)):[u] for u in V}
    sol = []
    for i in range(1,k+1):
        for u in V:
            dp_table[(u,i)]=set()
            for v in G[u]:
                for partial_coloring in dp_table[(v,i-1)]:
                    if coloring[u] not in partial_coloring:
                        new_partial_coloring = tuple(sorted(list(partial_coloring)+[coloring[u]]))
                        if not new_partial_coloring in dp_table[(u,i)]:
                            dp_table[(u,i)].add(new_partial_coloring)
                            partial_paths[(u, new_partial_coloring)]=[u] +partial_paths[v,partial_coloring]
                            if i==k-1:
                                print(partial_paths[(u,new_partial_coloring)])
                                sol.append(partial_paths[(u,new_partial_coloring)])
    # pprint(coloring)
    pprint(dp_table)
    # pprint(partial_paths)
    print("SOLUTION")
    print(sol)
    return sol

def main():
    if len(sys.argv)<3:
    # python k_path_dp.py data/test_example_list 3 random
        print("Usage "+ sys.argv[0]+ " <graph-file.edge> <k> <opt: coloring-type> <opt: coloring-at-run-time>")     #example file: "data/test_example_list"
    else:
        return DP_from_coloring(sys.argv[1], sys.argv[2],type = sys.argv[3])


if __name__=="__main__":
    main()