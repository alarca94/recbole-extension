import torch
from scipy.sparse import coo_matrix
import torch.nn.functional as F
import numpy as np

from scipy.sparse import csr_matrix


def scipy_vs_torch_sparse():
    def unique_per_row(m):
        m_sorted = torch.sort(m)[0]
        duplicates = m_sorted[:, 1:] == m_sorted[:, :-1]
        m_sorted[:, 1:][duplicates] = 0
        return m_sorted

    def data_masks2(all_sessions):
        unique_values = unique_per_row(all_sessions)
        indices = unique_values.nonzero(as_tuple=True)
        col_ixs = unique_values[indices] - 1
        # crow_ixs = F.pad((unique_values > 0).sum(1), pad=(1, 0), mode="constant", value=0)
        return torch.sparse_coo_tensor(torch.stack([indices[0], col_ixs]), torch.ones_like(col_ixs), size=(len(all_sessions), 5))

    def data_masks(all_sessions):
        indptr, indices, data = [], [], []
        indptr.append(0)
        for j in range(len(all_sessions)):
            session = np.unique(all_sessions[j])
            length = len(session)
            s = indptr[-1]
            indptr.append((s + length))
            for i in range(length):
                indices.append(session[i] - 1)
                data.append(1)

        print(f'data: {data}')
        print(f'indices: {indices}')
        print(f'indptr: {indptr}')
        matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), 5))
        print(matrix.todense())

    def data_masks_scipy(all_sessions):
        unique_values = unique_per_row(all_sessions)
        indices = unique_values.nonzero(as_tuple=True)
        col_ixs = unique_values[indices] - 1
        # crow_ixs neeeded for sparse_csr_tensor (not available in torch==1.8.1)
        crow_ixs = F.pad((unique_values > 0).sum(1), pad=(1, 0), mode="constant", value=0).cumsum(0)
        values = torch.ones_like(col_ixs).tolist()
        indices = col_ixs.tolist()
        indptr = crow_ixs.tolist()
        print(f'Values {values}')
        print(f'indices {indices}')
        print(f'indptr {indptr}')
        return csr_matrix((values, indices, indptr),
                          shape=(len(all_sessions), 5))

    s = [[4, 2, 1], [2, 4, 5, 1], [1, 3, 1]]
    data_masks(s)
    s2 = torch.tensor([[4, 2, 1, 0], [2, 4, 5, 1], [1, 3, 1, 0]])
    print(data_masks2(s2).to_dense())
    print('Hey')
    print(data_masks_scipy(s2).todense())


def test_hyperaugmentation():
    window_sizes = [1, 2, 3]
    item_seq = torch.tensor([[4, 4, 78, 1, 3, 0, 0, 0, 0, 0]])
    node_mapping = {78: 0, 4: 1, 1: 2, 3: 3}
    c_seq = [node_mapping[item] for item in item_seq[0, :5].detach().numpy()]
    max_n_node = 8

    u_A_in = [[], [], []]  # torch.zeros((max_n_node, n_edges))
    u_A_out = [[], [], []]  # torch.zeros((max_n_node, n_edges))
    it = 0
    for wsz in window_sizes:
        if len(c_seq) > wsz:
            for i in range(len(c_seq) - wsz):
                ixs, vals = np.unique(c_seq[i:i + wsz], return_counts=True)
                u_A_out[0].extend(ixs)
                u_A_out[1].extend([it] * len(ixs))
                u_A_out[2].extend(vals)
                ixs, vals = np.unique(c_seq[i + wsz:i + wsz * 2], return_counts=True)
                u_A_in[0].extend(ixs)
                u_A_in[1].extend([it] * len(ixs))
                u_A_in[2].extend(vals)
                it += 1
    print(f'The input sequence is {c_seq}')
    print(f'Subsets de salida: {list(zip(u_A_out[0], u_A_out[1], u_A_out[2]))}')
    print(f'Subsets de entrada: {list(zip(u_A_in[0], u_A_in[1], u_A_in[2]))}')

    u_A_out = coo_matrix((u_A_out[2], u_A_out[:2]), shape=(max_n_node, u_A_in[1][-1]+1))

    print(u_A_out.sum(1))
    print(f'A_OUT: {u_A_out.toarray()}')

