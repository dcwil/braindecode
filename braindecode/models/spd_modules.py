import numpy as np
import torch
from torch import nn
from braindecode.models.modules import Expression

from collections import OrderedDict

from .spd_functional import (
    LogEig_I,
    LogEig_DK,
    ModEig_DK,
    DK_ExpOp,
    _regularise_with_oas_pytorch,
    ComputeRieMean,
    ReEig_I,
    ReEig_DK,
    DK_MapBatch,
    _get_band_indices
)
from .spd_optim import StiefelParameter


class ChSpecConv(nn.Module):
    def __init__(self, n_chans, n_filters, filter_time_length):
        super(__class__, self).__init__()
        self.conv = nn.Conv1d(
            n_chans, n_filters * n_chans, filter_time_length, groups=n_chans
        )
        self.band_indices = _get_band_indices(n_chans, n_filters)

    def forward(self, X):
        X = self.conv(X)
        # reshape into bands
        X = torch.index_select(X, dim=1, index=self.band_indices)
        return X


class SpatConv(nn.Module):
    def __init__(self, n_in_chs, n_out_chs):
        super(__class__, self).__init__()
        self.conv = nn.Sequential(
            Expression(lambda x: x.unsqueeze(-1).transpose(1, 3)),
            nn.Conv2d(1, n_out_chs, (1, n_in_chs)),
            Expression(
                lambda x: x.transpose(2, 3).reshape(
                    x.shape[0], x.shape[1] * x.shape[3], x.shape[2]
                )
            ),
        )

    def forward(self, X):
        X = self.conv(X)

        return X


class SCMPool(nn.Module):
    def __init__(self, n_filters, regularise=False):
        super(__class__, self).__init__()
        self.n_filters = n_filters
        self.regularise = regularise

    def forward(self, X):
        batch_size, n_features, n_samps = X.shape
        # get actual n_chans
        n_chans = n_features // self.n_filters

        scm = (1 / (n_samps - 1)) * X.matmul(X.transpose(-1, -2))

        if self.regularise:
            scm = _regularise_with_oas_pytorch(scm, n_samps, n_features)

        return scm


class SPDVectorize(nn.Module):
    """Straight from https://github.com/adavoudi/spdnet/blob/master/spdnet/spd.py"""

    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        row_idx, col_idx = np.triu_indices(input_size)
        self.register_buffer("row_idx", torch.LongTensor(row_idx))
        self.register_buffer("col_idx", torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output


class LogEig(nn.Module):
    """Adapted from adavoudi/spdnet to include switching between backprops"""

    def __init__(self, mode="I"):
        super(LogEig, self).__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == "I":
            log = LogEig_I
        elif self.mode == "DK":
            log = LogEig_DK
        else:
            raise ValueError

        output = log.apply(x)

        return output


class ExpEig(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ModEig_DK.apply(x, DK_ExpOp)


class DK_LogMapBatchAtMean(nn.Module):
    """Do the logarithmic map to the batch using mean as reference point"""

    def __init__(self, mean):
        super().__init__()
        self.mean = mean

    def forward(self, x):
        if self.mean == "euc":
            x_mean = torch.mean(x, 0)
        elif self.mean == "rie":
            x_mean = ComputeRieMean(x)
        else:
            raise ValueError

        out = DK_MapBatch(x, x_mean, "log")
        return out


class AltReEig(nn.Module):
    def __init__(self, epsilon=1e-4, mode="I"):
        super(AltReEig, self).__init__()
        self.register_buffer("epsilon", torch.FloatTensor([epsilon]))
        self.mode = mode

    def forward(self, input):
        if self.mode == "I":
            re = ReEig_I
        elif self.mode == "DK":
            re = ReEig_DK
        else:
            raise ValueError

        output = re.apply(input, self.epsilon)
        return output


class BiMap(nn.Module):
    """Straight from https://github.com/adavoudi/spdnet/blob/master/spdnet/spd.py , but renamed from SPDTransform."""

    def __init__(self, input_size, output_size):
        super(BiMap, self).__init__()
        self.increase_dim = None
        if output_size > input_size:
            self.increase_dim = SPDIncreaseDim(input_size, output_size)
            input_size = output_size
        self.weight = StiefelParameter(
            torch.FloatTensor(input_size, output_size), requires_grad=True
        )
        nn.init.orthogonal_(self.weight)

    def forward(self, input):
        output = input
        if self.increase_dim:
            output = self.increase_dim(output)
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(input.size(0), -1, -1)
        output = torch.bmm(weight.transpose(1, 2), torch.bmm(output, weight))

        return output


class SPDIncreaseDim(nn.Module):
    """Straight from https://github.com/adavoudi/spdnet/blob/master/spdnet/spd.py """

    def __init__(self, input_size, output_size):
        super(SPDIncreaseDim, self).__init__()
        self.register_buffer('eye', torch.eye(output_size, input_size))
        add = np.asarray([0] * input_size + [1] * (output_size - input_size), dtype=np.float32)
        self.register_buffer('add', torch.from_numpy(np.diag(add)))

    def forward(self, input):
        eye = self.eye.unsqueeze(0)
        eye = eye.expand(input.size(0), -1, -1)
        add = self.add.unsqueeze(0)
        add = add.expand(input.size(0), -1, -1)

        output = torch.baddbmm(add, eye, torch.bmm(input, eye.transpose(1, 2)))

        return output


class EmbedMLP(nn.Module):
    def __init__(self, num_embeddings_dict, embed_dim, scale_grad_by_freq, out_dim, drop_prob=0.5, hidden_layer=False,
                 verbose=True):
        super(__class__, self).__init__()
        self.num_embeddings_dict = num_embeddings_dict
        self.embed_dim = embed_dim
        self.scale_grad_by_freq = scale_grad_by_freq
        self.drop_prob = drop_prob
        self.hidden_layer = hidden_layer

        self.embedding_layers = nn.ModuleDict({
            k: nn.Embedding(
                num_embeddings=n_emb,
                embedding_dim=embed_dim,
                scale_grad_by_freq=scale_grad_by_freq
            ) for k, n_emb in num_embeddings_dict.items()
        })
        # if verbose:
        # print(num_embeddings_dict)

        # len of num embeddings dict is the number of cat feats
        input_size = len(num_embeddings_dict) * embed_dim

        layer_sizes = torch.linspace(input_size, out_dim, steps=4 if hidden_layer else 3).long()

        mlp_dict = OrderedDict()
        mlp_dict['linear'] = nn.Linear(layer_sizes[0], layer_sizes[1])
        mlp_dict['dropout'] = nn.Dropout(p=drop_prob)
        mlp_dict['gelu'] = nn.GELU()

        if hidden_layer:
            mlp_dict['h_linear'] = nn.Linear(layer_sizes[1], layer_sizes[2])
            mlp_dict['h_dropout'] = nn.Dropout(p=drop_prob)
            mlp_dict['h_gelu'] = nn.GELU()

        mlp_dict['final_linear'] = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        self.mlp = nn.Sequential(mlp_dict)
        # self.embed_spd = EmbedSPD(layer_sizes[-1])

    def forward(self, categorical):
        # collect embedded feature vecs
        x = torch.stack([self.embedding_layers[k](v.long()) for k, v in categorical.items()],
                        dim=0)  # num embedding layers x batch size x embed_dim
        x = x.transpose(0, 1)  # batch x emb layers x emb dim
        x = x.flatten(start_dim=1, end_dim=-1)  # batch x (emb_layers * emb_dim)

        x = self.mlp(x)
        # x = self.embed_spd(x)

        return x