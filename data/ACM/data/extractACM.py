import torch
import pickle
import numpy as np
from scipy import sparse


with open('ACM3025.pkl', 'rb') as f:
    data = pickle.load(f)
f.close

labels, features = torch.from_numpy(data['label'].todense()).long(), torch.from_numpy(data['feature'].todense()).float()
num_nodes = data['label'].shape[0]
adjacents = []
adjacent1 = sparse.csr_matrix(data['PAP'] - np.eye(num_nodes)).toarray()
adjacent2 = sparse.csr_matrix(data['PLP'] - np.eye(num_nodes)).toarray()

adjacent1 = torch.from_numpy(adjacent1)
idx1 = torch.nonzero(adjacent1).T
data1 = adjacent1[idx1[0], idx1[1]]
adjacent1 = sparse.coo_matrix((data1, (idx1[0], idx1[1])), adjacent1.shape)
adjacents.append(adjacent1)

adjacent2 = torch.from_numpy(adjacent2)
idx2 = torch.nonzero(adjacent2).T
data2 = adjacent2[idx2[0], idx2[1]]
adjacent2 = sparse.coo_matrix((data2, (idx2[0], idx2[1])), adjacent2.shape)
adjacents.append(adjacent2)

with open('../labels.pickle', 'wb') as l:
    pickle.dump(labels, l)
l.close
with open('../features.pickle', 'wb') as f:
    pickle.dump(features, f)
f.close
with open('../adjacents.pickle', 'wb') as a:
    pickle.dump(adjacents, a)
a.close
