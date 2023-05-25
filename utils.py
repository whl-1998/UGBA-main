# %%
import torch
import numpy as np


def tensor2onehot(labels):
    """Convert label tensor to label onehot tensor.
    Parameters
    ----------
    labels : torch.LongTensor
        node labels
    Returns
    -------
    torch.LongTensor
        onehot labels tensor
    """
    labels = labels.long()
    eye = torch.eye(labels.max() + 1)
    onehot_mx = eye[labels]
    return onehot_mx.to(labels.device)


def accuracy(output, labels):
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def idx_to_mask(indices, n):
    mask = torch.zeros(n, dtype=torch.bool)
    mask[indices] = True
    return mask


import scipy.sparse as sp


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


# Returns the induced subgraph
def subgraph(subset, edge_index):
    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    return edge_index, edge_mask


# 我们会在测试集中选一半作为 attack-set，另一半作为 clean test-set
def get_split(args, data, device):
    random_state = np.random.RandomState(args.seed)
    # perm = random_state.permutation(data.num_nodes)  # get random idx of nodes
    perm = np.arange(data.num_nodes)
    train_number = int(0.2 * len(perm))  # train_set: 20%
    idx_train = torch.tensor(sorted(perm[:train_number]), dtype=torch.long).to(device)
    data.train_mask = torch.zeros_like(data.train_mask)  # if true, means this pos has train element
    data.train_mask[idx_train] = True

    val_number = int(0.1 * len(perm))
    idx_val = torch.tensor(sorted(perm[train_number: train_number + val_number]), dtype=torch.long).to(device)
    data.val_mask = torch.zeros_like(data.val_mask)
    data.val_mask[idx_val] = True

    test_number = int(0.2 * len(perm))
    idx_test = torch.tensor(sorted(perm[train_number + val_number: train_number + val_number + test_number]),
                            dtype=torch.long).to(device)
    data.test_mask = torch.zeros_like(data.test_mask)
    data.test_mask[idx_test] = True

    idx_clean_test = idx_test[: int(test_number / 2)]
    idx_atk = idx_test[int(test_number / 2):]
    return data, idx_train, idx_val, idx_clean_test, idx_atk
