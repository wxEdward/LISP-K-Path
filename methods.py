import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader,RandomNodeSampler, NeighborLoader
from torch_geometric.utils import structured_negative_sampling
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


#"data/ppi_edge_list_mapped.npy"
#"data/ppi_embeddings_features.npy"
def load(edge_list, embedding=None):
    edge_index= np.load(edge_list)
    if embedding:
        embeddings_features=np.load(embedding)
    #placeholder for the other embedding
    # else:
    #     embeddings_features = np.random()
    return edge_index,embeddings_features

def loss_cos_sim(output, sample):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = 0
    i, j, k = structured_negative_sampling(sample.edge_index)
    if output[i].argmax()==output[k].argmax():
        loss -=(cos(sample.x[i],sample.x[k])).sum()
    else:
        loss +=cos(sample.x[i],sample.x[k]).sum()
    if output[i].argmax()==output[j].argmax():
        loss +=cos(sample.x[i],sample.x[j]).sum()
    else:
        loss -=cos(sample.x[i],sample.x[j]).sum()
    return loss
        


class GCN(torch.nn.Module):
    def __init__(self,data):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 32)
        self.conv2 = GCNConv(32, 16)   #hm, could make # of color as the output dim here?

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #x = x.to(dtype=torch.float32)
        # print((x.dtype))
        # print((edge_index.dtype))
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
    def similarity(self, x, y, test=False):
      x, y = F.normalize(x), F.normalize(y)
      sim = x.mm(y.t())
      if test:
        # the testing code wants the index of the best match
        _, sim = sim.max(0)
      return sim

class GCN_class(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, data.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

def train_1(edge_list, embedding=None):
    edge_index,embeddings_features =load(edge_list,embedding)
    t = torch.from_numpy(edge_index)
    data = Data(x = torch.from_numpy(embeddings_features),edge_index=t.t().contiguous())
    loader = NeighborLoader(data, num_neighbors=[10] * 2, shuffle=True, batch_size=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    margin = 0.01
    train_running_loss = 0.0
    for sample in loader:
        # print(device)
        sample.x = sample.x.to(device).to(dtype=torch.float32)
        sample.x.requires_grad=True
        # sample.x = requires_grad=True
        sample.edge_index = sample.edge_index.to(device).long()
        # print(sample.x.is_cuda)
        # print(type(sample.edge_index))
        # print(sample.edge_index.is_cuda)
        optimizer.zero_grad()
        i, j, k = structured_negative_sampling(sample.edge_index)
        negatives = (i,k)   #not neighbors
        positives = (i,j)   #neighbors 
        output = model(sample)
        #pos = model.similarity(sample.x[i], sample.x[j])
        #neg = model.similarity(sample.x[i], sample.x[k])
        pos = model.similarity(output[i], output[j])
        neg = model.similarity(output[i], output[k])
        diff =pos.diag() -neg.diag() +margin      # Note for coloring, we want negatives closer and positives further
        triplet_loss_matrix = diff.mean()
        loss = triplet_loss_matrix
        loss.backward()
        optimizer.step()
        train_running_loss += loss.detach().item()
        
def train_2(edge_list, k,embedding=None):
    edge_index,embeddings_features =load(edge_list,embedding)
    t = torch.from_numpy(edge_index)
    data = Data(x = torch.from_numpy(embeddings_features),edge_index=t.t().contiguous())
    data.num_class=k
    loader = NeighborLoader(data, num_neighbors=[10] * 2, shuffle=True, batch_size=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(30):
        for sample in loader:
            sample.x = sample.x.to(device).to(dtype=torch.float32)
            sample.x.requires_grad=True
            sample.edge_index = sample.edge_index.to(device).long()
            optimizer.zero_grad()
            output = model(sample)
            sample.x = sample.x.to(device).to(dtype=torch.float32)
            sample.x.requires_grad=True
            loss = loss_cos_sim(output, sample)
            loss.backward()
            optimizer.step()
    return model, data
    

def predict(model,data):
    pred = model(data).argmax(dim=1)
    return pred


def method2(edge_file):
    #hard coding for now
    model,data = train_2(edge_file,k,embedding= "data/ppi_embeddings_features.npy")
    prediction = predict(model,data)
    coloring={}
    for i in range(len(data.x)):
        coloring[i]=prediction[i]
    return coloring
