import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.backbone

from utils.feature_align import feature_align
from module.transformer import CrossAttention


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)

def reverse_pad(mat, index):
    new_mat = torch.cat([mat[i][0: index[i]] for i in range(len(index))], dim=0)
    return new_mat

def pad_tensor(src_mat, n_mat):
    max_n_pt = max(n_mat)
    feat_dim = src_mat.size(1)
    res = []
    st = torch.tensor(0).cuda()
    for num in n_mat:
        res.append(torch.cat([src_mat[st : st + num], torch.zeros([max_n_pt-num, feat_dim]).cuda()], dim=0))
        st = st + num
    return  torch.stack(res, dim=0).cuda()


class SAL_layer(nn.Module):
    def __init__(self, num_heads):

        super(SAL_layer, self).__init__()
        self.num_heads = num_heads
        self.sal = nn.MultiheadAttention(1024, num_heads, 0.1)
    def forward(self, feat_A, feat_B):
        new_feat_A = self.sal(feat_A, feat_A, feat_A)[0]
        new_feat_B = self.sal(feat_B, feat_B, feat_B)[0]
        return new_feat_A, new_feat_B

class CAL_layer(nn.Module):
    def __init__(self, num_heads):
        super(CAL_layer, self).__init__()
        self.num_heads = num_heads
        self.cal = CrossAttention(1024, 0.1, num_heads)
    def forward(self, feat_A, feat_B):
        m_A, new_feat_A = self.cal(feat_A, feat_B, feat_B)
        m_B, new_feat_B = self.cal(feat_B, feat_A, feat_A)

        return m_A, m_B, new_feat_A, new_feat_B


class Net(utils.backbone.VGG16_bn):
    def __init__(self):
        super(Net, self).__init__()
        self.global_state_dim = 1024
        self.num_sal_heads = 8
        self.num_cal_heads = 8
        self.num_layers = 3
        self.position_embedding = nn.Sequential(nn.Linear(2,512),
                                                nn.ReLU(),
                                                nn.Linear(512, 1024),
                                                )
        for i in range(self.num_layers):
            self.add_module(f'sal_layer_{i}', SAL_layer(self.num_sal_heads))
            self.add_module(f'bn_sal_{i}', nn.BatchNorm1d(1024))
            self.add_module(f'cal_layer_{i}', CAL_layer(self.num_cal_heads))
            self.add_module(f'bn_cal_{i}', nn.BatchNorm1d(1024))


    def forward(self, images, points, n_points):

        graph_iter = 0
        feat_A = None
        feat_B = None
        m_A = None
        m_B = None
        for image, p, n_p in zip(images, points, n_points):

            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)

            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            Ut = concat_features(feature_align(nodes, p, n_p, (256, 256)), n_p)
            Ft = concat_features(feature_align(edges, p, n_p, (256, 256)), n_p)
            if graph_iter==0:
                feat_A = torch.cat((Ut, Ft), dim=-1)
                feat_A = pad_tensor(feat_A, n_p)
                graph_iter+=1
            else:
                feat_B = torch.cat((Ut, Ft), dim=-1)
                feat_B = pad_tensor(feat_B, n_p)

        points = [point/256 for point in points]
        feat_A = feat_A + self.position_embedding(points[0])
        feat_B = feat_B + self.position_embedding(points[1])

        for i in range(self.num_layers):
            sal_layer = getattr(self, f'sal_layer_{i}')
            new_feat_A, new_feat_B = sal_layer(feat_A, feat_B)
            # print('feat_A 1:',feat_A)
            feat_A = F.relu(feat_A + new_feat_A) + self.position_embedding(points[0])
            feat_B = F.relu(feat_B + new_feat_B) + self.position_embedding(points[1])

            bn_sal = getattr(self, f'bn_sal_{i}')
            feat_A = bn_sal(feat_A.transpose(1,2).contiguous()).transpose(1,2).contiguous()
            feat_B = bn_sal(feat_B.transpose(1,2).contiguous()).transpose(1,2).contiguous()

            # print('feat_A 2:',feat_A)
            cal_layer = getattr(self, f'cal_layer_{i}')
            m_A, m_B, new_feat_A, new_feat_B = cal_layer(feat_A, feat_B)
            feat_A = F.relu(feat_A + new_feat_A) + self.position_embedding(points[0])
            feat_B = F.relu(feat_B + new_feat_B) + self.position_embedding(points[1])
            # print('feat_A 3:', feat_A)
            bn_cal = getattr(self, f'bn_cal_{i}')
            feat_A = bn_cal(feat_A.transpose(1,2).contiguous()).transpose(1,2).contiguous()
            feat_B = bn_cal(feat_B.transpose(1,2).contiguous()).transpose(1,2).contiguous()
        # print('m_A: ', m_A, 'm_B:', m_B)

        res = torch.mean(m_A+m_B.transpose(2,3).contiguous(), dim=1)

        return res
