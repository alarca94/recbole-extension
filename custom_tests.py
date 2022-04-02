import time

import torch
from scipy.sparse import coo_matrix
import torch.nn.functional as F
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.linalg import eig

from scipy.sparse import csr_matrix, eye, hstack


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
    window_sizes = [1, 2]
    item_seq = torch.tensor([[4, 4, 78, 1, 3, 78, 0, 0, 0, 0], [4, 4, 78, 1, 3, 78, 0, 0, 0, 0]])
    node_mapping = {78: 0, 4: 1, 1: 2, 3: 3, 0: 4}

    seq = [[node_mapping[item] for item in item_seq[row].detach().numpy()] for row in range(item_seq.shape[0])]
    max_n_node = 8

    def regular_approach():
        c_seq = seq[0]
        start_time = time.time()
        h_A_in = [[], [], []]  # torch.zeros((max_n_node, n_edges))
        h_A_out = [[], [], []]  # torch.zeros((max_n_node, n_edges))
        it = 0
        for wsz in window_sizes:
            if len(c_seq) > wsz:
                for i in range(len(c_seq) - wsz):
                    ixs, vals = np.unique(c_seq[i:i + wsz], return_counts=True)
                    h_A_out[0].extend(ixs)
                    h_A_out[1].extend([it] * len(ixs))
                    h_A_out[2].extend(vals)
                    ixs, vals = np.unique(c_seq[i + wsz:i + wsz * 2], return_counts=True)
                    h_A_in[0].extend(ixs)
                    h_A_in[1].extend([it] * len(ixs))
                    h_A_in[2].extend(vals)
                    it += 1
            else:
                break

        h_A_out = coo_matrix((h_A_out[2], h_A_out[:2]), shape=(max_n_node, h_A_out[1][-1] + 1))
        h_A_in = coo_matrix((h_A_in[2], h_A_in[:2]), shape=(max_n_node, h_A_in[1][-1] + 1))
        print(f'Regular method {time.time() - start_time} s')
        invD_out = 1. / np.maximum(1., h_A_out.sum(1))
        invD_in = 1. / np.maximum(1., h_A_in.sum(1))
        invB_out = 1. / h_A_out.sum(0)
        invB_in = 1. / h_A_in.sum(0)

        u_A_out = h_A_in.multiply(invB_in).dot(h_A_out.multiply(invD_out).T)
        u_A_in = h_A_out.multiply(invB_out).dot(h_A_in.multiply(invD_in).T)
        u_A = np.concatenate([u_A_in.toarray(), u_A_out.toarray()]).T
        print(f'The input sequence is {c_seq}')
        # print(f'Subsets de salida: {list(zip(u_A_out[0], u_A_out[1], u_A_out[2]))}')
        # print(f'Subsets de entrada: {list(zip(u_A_in[0], u_A_in[1], u_A_in[2]))}')

        # u_A_out = coo_matrix((u_A_out[2], u_A_out[:2]), shape=(max_n_node, u_A_in[1][-1]+1))

        # print(u_A.sum(1))
        # print(f'A:\n{u_A}')
        # print(f'A:\n{h_A_out.dot(h_A_in.T).toarray().T}')
        # norm_h_out = h_A_out.multiply(invD_out).dot(h_A_out.multiply(invB_out).T)
        # norm_h_in = h_A_in.multiply(invD_in).dot(h_A_in.multiply(invB_in).T)
        # A = h_A_out.multiply(invD_out).dot(h_A_in.multiply(invB_in).T).toarray().T
        # print(eig(A)[0].max())
        # print(f'A:\n{A}')


    # TODO: Temporal decay (F-Hypergraph)
    # TODO: Backward attention (B-Hypergraph)

    def efficient_approach():
        start_time = time.time()
        def sliding_window(seq, wsz, sep):
            shape = seq.shape[:-1] + (seq.shape[-1] - wsz + 1, wsz)
            strides = seq.strides + (seq.strides[-1],)
            windows = as_strided(seq, shape=shape, strides=strides)
            cols = np.repeat(np.arange(shape[-2]-wsz), wsz)
            src_win = coo_matrix((np.repeat([1], len(cols)), (windows[:, :-wsz].ravel(), cols)), shape=(max_n_node, cols[-1]+1))
            trg_win = coo_matrix((np.repeat([1], len(cols)), (windows[:, (wsz+sep):].ravel(), cols)), shape=(max_n_node, cols[-1]+1))
            return src_win, trg_win
        # print(item_seq)
        eb = np.pad(seq, ((0, 0), [window_sizes[-1] - 1] * 2), constant_values=node_mapping[0])
        windows = list(zip(*[sliding_window(eb, wsz, 0) for wsz in window_sizes]))
        Ht = hstack(windows[0]).tocsr()
        Hh = hstack(windows[1]).tocsr()
        Ht[node_mapping[0], :] = 0
        Hh[node_mapping[0], :] = 0
        Ht[:, node_mapping[0]] = 0
        Hh[:, node_mapping[0]] = 0
        print(f'Efficient method {time.time() - start_time} s')
        print('hey')

    regular_approach()
    efficient_approach()

    print('Hey')


def test_oriented_graphs():
    s = [0, 1, 2, 0, 1, 0]
    h = np.zeros((len(np.unique(s)), len(s) - (2 - 1)))


if __name__ == '__main__':
    test_hyperaugmentation()
