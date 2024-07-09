import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from nilearn.connectome import ConnectivityMeasure
from signed_gcn import signedGCN


def get_data_loader_from_matrices(matrices_list, x, threshold):
  # Create a list of PyG Data objects
  data_list = []
  for adj in matrices_list:
      edge_index, edge_weight = dense_to_sparse(torch.from_numpy(adj))
      data = Data(x = x, edge_index=edge_index, edge_weight = edge_weight)
      data_list.append(data)
  # Create a DataLoader
  loader = DataLoader(data_list, batch_size=2, shuffle=True)
  # Iterate through the DataLoader
  return loader



cov_measure = ConnectivityMeasure(
    kind="covariance",
    standardize="zscore_sample",
)

X = torch.tensor(np.load("Data/X.npy"))
y = torch.tensor(np.load("Data/y.npy"))

cov_matrices = cov_measure.fit_transform(X)
train_loader = get_data_loader_from_matrices(cov_matrices, 0.1)

num_classes = 2
gcn_model = signedGCN(X.shape[2], num_classes)

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01)

gcn_model.train()
for epoch in range(50):
    losses = 0
    for data in train_loader:
        # data = data.to(“cuda”)
        optimizer.zero_grad()
        out = gcn_model(data)
        loss = criterion(out, data.y)
        losses += loss.item()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {losses/len(train_loader)}")




