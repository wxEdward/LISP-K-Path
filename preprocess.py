import csv
import numpy as np
import pandas as pd
import torch
import sys


def readData(filename,undirected=True):
    raw_data = pd.read_csv(filename)
    if undirected:
        rest = raw_data.copy()
        rest = rest.rename(columns={0: 1, 1: 0})
        full_data =raw_data.append(rest)
    else:
        full_data=raw_data
    np.savetxt(filename+'_edge_list', full_data.values, fmt='%d')


def readDeepWalk(embedding):
    with open(embedding,'r') as f:
        embeddings = f.read().splitlines()
    embeddings_input = pd.read_table('ppi.embeddings',sep=" ",skiprows=[0],header=None)
    embeddings_input = embeddings_input.sort_values(by=[0])  #sort by node number
    embeddings = embeddings_input.to_numpy()
    embeddings_features=embeddings[:,1:]    # Splice to have features only (first column is node names)
    np.save(embedding+"_features", embeddings_features, fmt='%d')


def mapEdgeNodesCont(edge_list):
    edge_index = pd.read_table(edge_list,sep=" ").to_numpy()
    all_nodes = np.unique(edge_index.T[0])   # We just need to find all unique occurences in one column only since the matrix contains undirected entries (each edge appears twice in reversed direction)
    dict_map = {k: v for v, k in enumerate(all_nodes)}
    for i in range(edge_index.shape[0]):
        for j in range(edge_index.shape[1]):
            edge_index[i,j] = dict_map[edge_index[i,j]]
    np.save(edge_list+"_mapped", edge_index, fmt='%d')


def main():
    if len(sys.argv)<2:
    # python preprocess.py data/PP-Pathways_ppi.csv data/ppi.embeddings
        print("Usage "+ sys.argv[0]+ "<data file> <embeddings>")
    else:
        readData(sys.argv[2])
        readDeepWalk(sys.argv[3])
        mapEdgeNodesCont(sys.argv[2]+'_edge_list')


if __name__=="__main__":
    main()
