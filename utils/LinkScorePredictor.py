import torch.nn as nn
import dgl
import dgl.function as fn


class LinkScorePredictor(nn.Module):
    """
    a single layer Edge Score Predictor
    """
    def __init__(self, hid_dim):
        super(LinkScorePredictor, self).__init__()

        self.projection_layer = nn.Linear(hid_dim, hid_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, edge_subgraph: dgl.DGLHeteroGraph, nodes_representation: dict, etype: str):
        """

        :param edge_subgraph: sampled subgraph
        :param nodes_representation: input node features, dict
        :param etype: predict edge type, str
        :return: scores, dst_node features
        """

        edge_subgraph = edge_subgraph.local_var()
        edge_type_subgraph = edge_subgraph[etype]
        for ntype in nodes_representation:
            edge_type_subgraph.nodes[ntype].data['h'] = self.projection_layer(nodes_representation[ntype])
        
        edge_type_subgraph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)

        # features for recommendation evlauation
        edge_type_subgraph.apply_edges(lambda edges: {'dst_feat': edges.dst['h']})
        edge_type_subgraph.apply_edges(lambda edges: {'src_id': edges.src['_ID']})
        edge_type_subgraph.apply_edges(lambda edges: {'dst_id': edges.dst['_ID']})
        
        return self.sigmoid(edge_type_subgraph.edata['score']), edge_type_subgraph.edata['dst_feat'], edge_type_subgraph.edata['src_id'], edge_type_subgraph.edata['dst_id']
