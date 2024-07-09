import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from nilearn.connectome import ConnectivityMeasure
from signed_gcn import signedGCN
from sklearn.model_selection import train_test_split


def get_data_loader_from_matrices(matrices_list, x, threshold, y):
  # Create a list of PyG Data objects
  data_list = []
  for i, adj in enumerate(matrices_list):
      edge_index, edge_weight = dense_to_sparse(torch.from_numpy(adj))
      data = Data(x = x[i].float(), edge_index=edge_index, 
      edge_weight = edge_weight.float(), y=y[i])
      data_list.append(data)
  # Create a DataLoader
  loader = DataLoader(data_list, batch_size=50, shuffle=True)
  # Iterate through the DataLoader
  return loader



cov_measure = ConnectivityMeasure(
    kind="covariance",
    standardize="zscore_sample",
)

X = np.load("Data/X.npy")
y = torch.LongTensor(np.load("Data/y.npy"))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

cov_matrices_train = cov_measure.fit_transform(X_train)
cov_matrices_test = cov_measure.fit_transform(X_test)

train_loader = get_data_loader_from_matrices(cov_matrices_train, 
torch.FloatTensor(X_train).permute((0,2,1)), 0.1, y_train)

test_loader = get_data_loader_from_matrices(cov_matrices_test, 
torch.FloatTensor(X_test).permute((0,2,1)), 0.1, y_test)

num_classes = 2
print(X.shape)
gcn_model = signedGCN(X.shape[1], 24, num_classes)

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([2.34848485, 0.6352459]))
optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01)

for epoch in range(100):
    gcn_model.train()
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
        
        gcn_model.eval()
        correct = 0

        for data in test_loader:
            out = gcn_model(data)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        print("Accuracy: ", correct / len(test_loader.dataset))



gcn_model.eval()
correct = 0
for data in test_loader:
    out = gcn_model(data)
    pred = out.max(dim=1)[1]
    correct += pred.eq(data.y).sum().item()
print("Accuracy: ", correct / len(test_loader.dataset))






for data in test_loader:
    # data = data.to(“cuda”)
    out = gcn_model(data)
    



