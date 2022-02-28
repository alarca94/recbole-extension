import logging
import torch as th
import dgl
import dgl.ops as F
import pandas as pd
import numpy as np

from dgl.nn.pytorch import edge_softmax
from collections import defaultdict, Counter
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class HomoAttentionAggregationLayer(nn.Module):
    def __init__(
        self,
        qry_feats,
        key_feats,
        val_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        activation=None,
        batch_norm=True,
    ):
        super().__init__()
        if batch_norm:
            self.batch_norm_q = nn.BatchNorm1d(qry_feats)
            self.batch_norm_k = nn.BatchNorm1d(key_feats)
        else:
            self.batch_norm_q = None
            self.batch_norm_k = None

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.fc_q = nn.Linear(qry_feats, val_feats, bias=True)
        self.fc_k = nn.Linear(key_feats, val_feats, bias=False)
        self.fc_v = nn.Linear(qry_feats, val_feats, bias=False)
        self.attn_e = nn.Parameter(
            th.randn(1, val_feats, dtype=th.float), requires_grad=True
        )
        self.activation = activation

        self.val_feats = val_feats
        self.num_heads = num_heads
        self.head_feats = val_feats // num_heads

    def extra_repr(self):
        return '\n'.join([
            f'num_heads={self.num_heads}', f'(attn_e): Parameter(1, {self.val_feats})'
        ])

    def forward(self, g, ft_q, ft_k, ft_e=None, return_ev=False):
        if self.batch_norm_q is not None:
            ft_q = self.batch_norm_q(ft_q)
            ft_k = self.batch_norm_k(ft_k)
        q = self.fc_q(self.feat_drop(ft_q))
        k = self.fc_k(self.feat_drop(ft_k))
        v = self.fc_v(self.feat_drop(ft_q)).view(-1, self.num_heads, self.head_feats)
        e = F.u_add_v(g, q, k)
        if ft_e is not None:
            e = e + ft_e
        e = (self.attn_e * th.sigmoid(e)).view(-1, self.num_heads, self.head_feats).sum(
            -1, keepdim=True
        )
        if return_ev:
            return e, v
        a = self.attn_drop(edge_softmax(g, e))
        rst = F.u_mul_e_sum(g, v, a).view(-1, self.val_feats)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


class HeteroAttentionAggregationLayer(nn.Module):
    def __init__(
        self,
        kg,
        embedding_dim,
        num_heads=1,
        batch_norm=True,
        feat_drop=0.0,
        relu=False,
    ):
        super().__init__()
        self.batch_norm = nn.ModuleDict() if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.edge_aggregate = nn.ModuleDict()
        self.edge_embedding = nn.ModuleDict()
        self.linear_agg = nn.ModuleDict()
        self.linear_self = nn.ModuleDict()
        self.activation = nn.ModuleDict()
        self.vtype2eutypes = defaultdict(list)
        for utype, etype, vtype in kg.canonical_etypes:
            self.edge_aggregate[etype] = HomoAttentionAggregationLayer(
                embedding_dim,
                embedding_dim,
                embedding_dim,
                num_heads=num_heads,
                batch_norm=False,
                feat_drop=0.0,
                activation=None,
            )
            if 'cnt' in kg.edges[etype].data:
                num_cnt_embeddings = kg.edges[etype].data['cnt'].max() + 1
                self.edge_embedding[etype] = nn.Embedding(
                    num_cnt_embeddings, embedding_dim
                )
            self.vtype2eutypes[vtype].append((etype, utype))
        for vtype in self.vtype2eutypes:
            self.linear_agg[vtype] = nn.Linear(embedding_dim, embedding_dim, bias=True)
            self.linear_self[vtype] = nn.Linear(
                embedding_dim, embedding_dim, bias=False
            )
            self.activation[vtype] = nn.ReLU() if relu else nn.PReLU(embedding_dim)
        if self.batch_norm is not None:
            self.batch_norm.update({
                vtype: nn.BatchNorm1d(embedding_dim)
                for vtype in self.vtype2eutypes
            })

    def forward(self, g, ft_src):
        if self.batch_norm is not None:
            ft_src = {ntype: self.batch_norm[ntype](ft) for ntype, ft in ft_src.items()}
        if self.feat_drop is not None:
            ft_src = {ntype: self.feat_drop(ft) for ntype, ft in ft_src.items()}
        device = next(iter(ft_src.values())).device
        ft_dst = {
            vtype: ft_src[vtype][:g.number_of_dst_nodes(vtype)]
            for vtype in g.dsttypes
        }
        feats = {}
        for vtype, eutypes in self.vtype2eutypes.items():
            src_nid = []
            dst_nid = []
            num_utypes_nodes = 0
            src_val = []
            attn_score = []
            for etype, utype in eutypes:
                sg = g[etype]
                ft_e = (
                    self.edge_embedding[etype](sg.edata['cnt'].to(device))
                    if etype in self.edge_embedding else None
                )
                e, v = self.edge_aggregate[etype](
                    sg,
                    ft_src[utype],
                    ft_dst[vtype],
                    ft_e=ft_e,
                    return_ev=True,
                )
                uid, vid = sg.all_edges(form='uv', order='eid')
                src_nid.append(uid + num_utypes_nodes)
                dst_nid.append(vid)
                num_utypes_nodes += sg.number_of_src_nodes()
                src_val.append(v)
                attn_score.append(e)
            src_nid = th.cat(src_nid, dim=0)
            dst_nid = th.cat(dst_nid, dim=0)
            edge_softmax_g = dgl.heterograph(
                data_dict={('utypes', 'etypes', 'vtype'): (src_nid, dst_nid)},
                num_nodes_dict={
                    'utypes': num_utypes_nodes,
                    'vtype': g.number_of_dst_nodes(vtype)
                },
                device=device
            )
            src_val = th.cat(src_val, dim=0)  # (num_utypes_nodes, num_heads, num_feats)
            attn_score = th.cat(attn_score, dim=0)  # (num_edges, num_heads, 1)
            attn_weight = F.edge_softmax(edge_softmax_g, attn_score)
            agg = F.u_mul_e_sum(edge_softmax_g, src_val, attn_weight)
            agg = agg.view(g.number_of_dst_nodes(vtype), -1)
            feats[vtype] = self.activation[vtype](
                self.linear_agg[vtype](agg) + self.linear_self[vtype](ft_dst[vtype])
            )

        return feats


class KnowledgeGraphEmbeddingLayer(nn.Module):
    def __init__(
        self,
        knowledge_graph,
        node_feats,
        num_layers,
        residual=True,
        batch_norm=True,
        feat_drop=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            HeteroAttentionAggregationLayer(
                knowledge_graph,
                node_feats,
                batch_norm=batch_norm,
                feat_drop=feat_drop,
            ) for _ in range(num_layers)
        ])
        self.residual = residual

    def forward(self, graphs, feats):
        for layer, g in zip(self.layers, graphs):
            out_feats = layer(g, feats)
            if self.residual:
                feats = {
                    ntype: out_feats[ntype] + feat[:len(out_feats[ntype])]
                    for ntype, feat in feats.items()
                }
            else:
                feats = out_feats
        return feats


class SEFrame(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        knowledge_graph,
        num_layers,
        batch_norm=True,
        feat_drop=0.0,
        **kwargs,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim, max_norm=1)
        self.user_indices = nn.Parameter(
            th.arange(num_users, dtype=th.long), requires_grad=False
        )
        self.item_embedding = nn.Embedding(num_items, embedding_dim, max_norm=1)
        self.item_indices = nn.Parameter(
            th.arange(num_items, dtype=th.long), requires_grad=False
        )
        self.knowledge_graph = knowledge_graph
        self.KGE_layer = KnowledgeGraphEmbeddingLayer(
            knowledge_graph,
            embedding_dim,
            num_layers,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
        )

    def precompute_KG_embeddings(self):
        self.eval()
        kg_device = self.knowledge_graph.device
        ft_device = self.user_indices.device
        if kg_device != ft_device:
            logging.debug(f'Copying knowledge graph from {kg_device} to {ft_device}')
            self.knowledge_graph = self.knowledge_graph.to(ft_device)
        with th.no_grad():
            graphs = [self.knowledge_graph] * len(self.KGE_layer.layers)
            feats = {
                'user': self.user_embedding(self.user_indices),
                'item': self.item_embedding(self.item_indices),
            }
            self.KG_embeddings = self.KGE_layer(graphs, feats)

    def forward(self, inputs):
        if inputs is None:
            return self.KG_embeddings
        else:
            graphs, used_nodes = inputs
            feats = {
                'user': self.user_embedding(used_nodes['user']),
                'item': self.item_embedding(used_nodes['item']),
            }
            return self.KGE_layer(graphs, feats)


class UpdateCell(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.x2i = nn.Linear(input_dim, 2 * output_dim, bias=True)
        self.h2h = nn.Linear(output_dim, 2 * output_dim, bias=False)

    def forward(self, x, hidden):
        i_i, i_n = self.x2i(x).chunk(2, 1)
        h_i, h_n = self.h2h(hidden).chunk(2, 1)
        input_gate = th.sigmoid(i_i + h_i)
        new_gate = th.tanh(i_n + h_n)
        return new_gate + input_gate * (hidden - new_gate)


class PWGGNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_steps=1,
        batch_norm=True,
        feat_drop=0.0,
        activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.fc_i2h = nn.Linear(
            input_dim, hidden_dim, bias=False
        ) if input_dim != hidden_dim else None
        self.fc_in = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # self.upd_cell = nn.GRUCell(2 * hidden_dim, hidden_dim)
        self.upd_cell = UpdateCell(2 * hidden_dim, hidden_dim)
        self.fc_h2o = nn.Linear(
            hidden_dim, output_dim, bias=False
        ) if hidden_dim != output_dim else None
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.activation = activation

    def propagate(self, g, rg, feat):
        if g.number_of_edges() > 0:
            feat_in = self.fc_in(feat)
            feat_out = self.fc_out(feat)
            a_in = F.u_mul_e_sum(g, feat_in, g.edata['iw'])
            a_out = F.u_mul_e_sum(rg, feat_out, rg.edata['ow'])
            # a: (num_nodes, 2 * hidden_dim)
            a = th.cat((a_in, a_out), dim=1)
        else:
            num_nodes = g.number_of_nodes()
            a = feat.new_zeros((num_nodes, 2 * self.hidden_dim))
        hn = self.upd_cell(a, feat)
        return hn

    def forward(self, g, rg, feat):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        if self.feat_drop is not None:
            feat = self.feat_drop(feat)
        if self.fc_i2h is not None:
            feat = self.fc_i2h(feat)
        for _ in range(self.num_steps):
            feat = self.propagate(g, rg, feat)
        if self.fc_h2o is not None:
            feat = self.fc_h2o(feat)
        if self.activation is not None:
            feat = self.activation(feat)
        return feat


class PAttentionReadout(nn.Module):
    def __init__(self, embedding_dim, batch_norm=False, feat_drop=0.0, activation=None):
        super().__init__()
        if batch_norm:
            self.batch_norm = nn.ModuleDict({
                'user': nn.BatchNorm1d(embedding_dim),
                'item': nn.BatchNorm1d(embedding_dim)
            })
        else:
            self.batch_norm = None
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.fc_user = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.fc_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_last = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_e = nn.Linear(embedding_dim, 1, bias=False)
        self.activation = activation

    def forward(self, g, feat_i, feat_u, last_nodes):
        if self.batch_norm is not None:
            feat_i = self.batch_norm['item'](feat_i)
            feat_u = self.batch_norm['user'](feat_u)
        if self.feat_drop is not None:
            feat_i = self.feat_drop(feat_i)
            feat_u = self.feat_drop(feat_u)
        feat_val = feat_i
        feat_key = self.fc_key(feat_i)
        feat_u = self.fc_user(feat_u)
        feat_last = self.fc_last(feat_i[last_nodes])
        feat_qry = dgl.broadcast_nodes(g, feat_u + feat_last)
        e = self.fc_e(th.sigmoid(feat_qry + feat_key))  # (num_nodes, 1)
        e = e + g.ndata['cnt'].log().view_as(e)
        alpha = F.segment.segment_softmax(g.batch_num_nodes(), e)
        rst = F.segment.segment_reduce(g.batch_num_nodes(), alpha * feat_val, 'sum')
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


class SERecLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_steps=1,
        batch_norm=True,
        feat_drop=0.0,
        relu=False,
    ):
        super().__init__()
        self.fc_i = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_u = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.pwggnn = PWGGNN(
            embedding_dim,
            embedding_dim,
            embedding_dim,
            num_steps=num_steps,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            activation=nn.ReLU() if relu else nn.PReLU(embedding_dim),
        )
        self.readout = PAttentionReadout(
            embedding_dim,
            batch_norm=batch_norm,
            feat_drop=feat_drop,
            activation=nn.ReLU() if relu else nn.PReLU(embedding_dim),
        )

    def forward(self, g, feat, feat_u):
        rg = dgl.reverse(g, False, False)
        if g.number_of_edges() > 0:
            edge_weight = g.edata['w']
            in_deg = F.copy_e_sum(g, edge_weight)
            g.edata['iw'] = F.e_div_v(g, edge_weight, in_deg)
            out_deg = F.copy_e_sum(rg, edge_weight)
            rg.edata['ow'] = F.e_div_v(rg, edge_weight, out_deg)

        feat = self.pwggnn(g, rg, feat)
        last_nodes = g.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        ct_l = feat[last_nodes]
        ct_g = self.readout(g, feat, feat_u, last_nodes)
        sr = th.cat((ct_l, ct_g), dim=1)
        return sr


class SERec(SequentialRecommender):
    input_type = InputType.POINTWISE

    def __init__(
        self,
        config,
        dataset
    ):
        super(SERec, self).__init__(config, dataset)

        self.n_users = dataset.num(self.USER_ID)
        self.embedding_dim = config['embedding_dim']
        self.n_layers = config['n_layers']
        self.b_norm = config['batch_norm']
        self.feat_drop = config['feat_drop']
        self.relu = config['relu']

        social_network = read_social_network(dataset.dataset_path / 'edges.txt')
        self.knowledge_graph = build_knowledge_graph(dataset, social_network)

        self.seframe = SEFrame(
            self.n_users,
            self.n_items,
            self.embedding_dim,
            self.knowledge_graph,
            self.num_layers,
            batch_norm=self.b_norm,
            feat_drop=self.feat_drop,
        )
        self.fc_i = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.fc_u = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.PSE_layer = SERecLayer(
            self.embedding_dim,
            num_steps=1,
            batch_norm=self.b_norm,
            feat_drop=self.feat_drop,
            relu=self.relu,
        )
        input_dim = 3 * self.embedding_dim
        self.batch_norm = nn.BatchNorm1d(input_dim) if self.b_norm else None
        self.fc_sr = nn.Linear(input_dim, self.embedding_dim, bias=False)

    def forward(self, inputs, extra_inputs=None):
        KG_embeddings = self.seframe.forward(extra_inputs)

        uid, g = inputs
        iid = g.ndata['iid']  # (num_nodes,)
        feat_i = KG_embeddings['item'][iid]
        feat_u = KG_embeddings['user'][uid]
        feat = self.fc_i(feat_i) + dgl.broadcast_nodes(g, self.fc_u(feat_u))
        feat_i = self.PSE_layer(g, feat, feat_u)
        sr = th.cat([feat_i, feat_u], dim=1)
        if self.batch_norm is not None:
            sr = self.batch_norm(sr)
        logits = self.fc_sr(sr) @ self.item_embedding(self.item_indices).t()
        return logits


def read_social_network(csv_file):
    df = pd.read_csv(csv_file, sep='\t')
    g = dgl.graph((df.followee.values, df.follower.values))
    return g


def build_knowledge_graph(df_train, social_network, do_count_clipping=True):
    print('building heterogeneous knowledge graph...')
    followed_edges = social_network.edges()
    clicks = Counter()
    transits = Counter()
    for _, row in df_train.iterrows():
        uid = row['userId']
        seq = row['items']
        for iid in seq:
            clicks[(uid, iid)] += 1
        transits.update(zip(seq, seq[1:]))
    clicks_u, clicks_i = zip(*clicks.keys())
    prev_i, next_i = zip(*transits.keys())
    kg = dgl.heterograph({
        ('user', 'followedby', 'user'): followed_edges,
        ('user', 'clicks', 'item'): (clicks_u, clicks_i),
        ('item', 'clickedby', 'user'): (clicks_i, clicks_u),
        ('item', 'transitsto', 'item'): (prev_i, next_i),
    })
    click_cnts = np.array(list(clicks.values()))
    transit_cnts = np.array(list(transits.values()))
    if do_count_clipping:
        click_cnts = clip_counts(click_cnts)
        transit_cnts = clip_counts(transit_cnts)
    click_cnts = th.LongTensor(click_cnts) - 1
    transit_cnts = th.LongTensor(transit_cnts) - 1
    kg.edges['clicks'].data['cnt'] = click_cnts
    kg.edges['clickedby'].data['cnt'] = click_cnts
    kg.edges['transitsto'].data['cnt'] = transit_cnts
    return kg


def find_max_count(counts):
    max_cnt = np.max(counts)
    density = np.histogram(
        counts, bins=np.arange(1, max_cnt + 2), range=(1, max_cnt + 1), density=True
    )[0]
    cdf = np.cumsum(density)
    for i in range(max_cnt):
        if cdf[i] > 0.95:
            return i + 1
    return max_cnt


def clip_counts(counts):
    """
    Truncate the counts to the maximum value of the smallest 95% counts.
    This could avoid outliers and reduce the number of count embeddings.
    """
    max_cnt = find_max_count(counts)
    counts = np.minimum(counts, max_cnt)
    return counts