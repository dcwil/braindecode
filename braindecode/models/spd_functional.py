from dataclasses import dataclass

import torch
import numpy as np
from torch import nn
from torch.autograd import Function

from .util import symmetric, isclose


class LogEig_I(Function):
    """Copies https://github.com/adavoudi/spdnet/blob/2a15e908634cd8db6c75ea45d9e3bd567203eccf/spdnet/spd.py#L126-L175
    for Ionescu method.
    Adapts https://gitlab.lip6.fr/schwander/torchspdnet/-/blob/master/torchspdnet/functional.py for DK method
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s.log_()
            output[k] = u.mm(s.diag().mm(u.t()))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1)
            eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(1))
            for k, dx in enumerate(grad_output):
                x = input[k]
                u, s, v = x.svd()

                dx = symmetric(dx)

                s_log_diag = s.log().diag()
                s_inv_diag = (1 / s).diag()

                dLdV = 2 * (dx.mm(u.mm(s_log_diag)))
                # dLdV = 2 * torch.einsum('ij, jk, li -> lk', u, s_log_diag, g)
                dLdS = eye * (s_inv_diag.mm(u.t().mm(dx.mm(u))))

                P = calculate_P(s, mode="I")

                grad_input[k] = u.mm(symmetric(P.t() * (u.t().mm(dLdV))) + dLdS).mm(
                    u.t()
                )

        return grad_input


class LogEig_DK(Function):
    """Copies https://github.com/adavoudi/spdnet/blob/2a15e908634cd8db6c75ea45d9e3bd567203eccf/spdnet/spd.py#L126-L175
    for Ionescu method. Adapts https://gitlab.lip6.fr/schwander/torchspdnet/-/blob/master/torchspdnet/functional.py for DK method
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s.log_()
            output[k] = u.mm(s.diag().mm(u.t()))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            # eye = input.new(input.size(1))
            # eye.fill_(1);
            # eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(1))
            for k, dx in enumerate(grad_output):
                x = input[k]
                u, s, v = x.svd()

                P = calculate_P(s, mode="DK", DK_mod_op=DK_LogOp)

                dLdx = torch.einsum("ji, jk, kl, il, mi, nl -> mn", u, dx, u, P, u, u)
                grad_input[k] = dLdx

        return grad_input


class ModEig_DK(Function):
    @staticmethod
    def forward(ctx, x, DK_mod_op):
        out = x.new(x.size(0), x.size(1), x.size(2))

        for k, m in enumerate(x):
            u, s, v = m.svd()
            f_of_s = DK_mod_op.f_of_S(s)

            out[k] = u.mm(f_of_s.diag().mm(u.t()))

        ctx.save_for_backward(x)
        ctx.DK_mod_op = DK_mod_op
        return out

    @staticmethod
    def backward(ctx, dx):
        (x,) = ctx.saved_tensors
        DK_mod_op = ctx.DK_mod_op
        # print(x.shape)
        # x = x[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = x.new(x.size(0), x.size(1), x.size(2))

            for k, dm in enumerate(dx):
                m = x[k]
                u, s, v = m.svd()  # could save this and speed thigns up?

                P = calculate_P(s, mode="DK", DK_mod_op=DK_mod_op)

                dLdx = torch.einsum("ji, jk, kl, il, mi, nl -> mn", u, dm, u, P, u, u)
                grad_input[k] = dLdx

        return grad_input, None


def DK_MapBatch(x, ref, direction):
    """Map batch to tangent space (`direction='log'`) from tangent space (`direction='exp'`)"""

    if len(ref.shape) == 2:
        ref = ref.unsqueeze(0)

    ref_sqrt = ModEig_DK.apply(ref, DK_SqM)[0]
    ref_sqrt_inv = ModEig_DK.apply(ref, DK_SqM_inv)[0]

    # inner congruence
    inner = torch.einsum("ij, bjk, kl -> bil", ref_sqrt_inv, x, ref_sqrt_inv)

    if direction == "log":
        # log
        inner = ModEig_DK.apply(inner, DK_LogOp)
    elif direction == "exp":
        inner = ModEig_DK.apply(inner, DK_ExpOp)
    else:
        raise ValueError

    # outer congruence
    out = torch.einsum("ij, bjk, kl -> bil", ref_sqrt, inner, ref_sqrt)

    return out


def ComputeRieMean(x):
    """Compute the Riemannian mean using karcher flow with a single step"""

    x_mean = torch.mean(x, 0)  # euc mean across batch
    x_ts = DK_MapBatch(x, x_mean, "log")  # map batch to tangent space at euc mean
    x_ts_mean = torch.mean(x_ts, 0)  # `rie` mean in ts
    out = DK_MapBatch(
        x_ts_mean.unsqueeze(0), x_mean, "exp"
    )  # map back to manifold at euc mean
    return out


def calculate_P(s, mode="I", DK_mod_op=None):
    """Called G in Engin et al. and called L in torchspdnet library"""

    s = s.squeeze()
    S = s.unsqueeze(1)
    S = S.expand(-1, S.size(0))

    if mode == "I":
        # Eq 14 Huang et al.
        P = S - torch.einsum("ij -> ji", S)
        mask_zero = torch.abs(P) == 0
        P = 1 / P
        P[mask_zero] = 0
    elif mode == "DK":
        assert issubclass(DK_mod_op, DK_ModOp)


        f_of_S = DK_mod_op.f_of_S(S)
        df_of_S = DK_mod_op.df_of_S(S)

        # Eq 12 Engin et al.
        P = (f_of_S - torch.einsum("ij -> ji", f_of_S)) / (
            S - torch.einsum("ij -> ji", S)
        )

        P[P == -np.inf] = 0
        P[P == np.inf] = 0
        P[torch.isnan(P)] = 0

        df_of_S = df_of_S * torch.eye(s.shape[0])  # turn into diag matrix

        P = P + df_of_S
    else:
        raise ValueError

    return P


@dataclass
class DK_ModOp:
    @staticmethod
    def f_of_S(S):
        # for P_ij where i == j:
        # f_of_S(S_i) - f_of_S(S_j) / (S_i - S_j)
        pass

    @staticmethod
    def df_of_S(S):
        # for P_ij where i =/= j:
        # df_of_S(S)
        pass


@dataclass
class DK_LogOp(DK_ModOp):
    @staticmethod
    def f_of_S(S):
        return S.log()

    @staticmethod
    def df_of_S(S):
        return 1 / S


@dataclass
class DK_ExpOp(DK_ModOp):
    @staticmethod
    def f_of_S(S):
        return torch.exp(S)

    @staticmethod
    def df_of_S(S):
        return torch.exp(S)


@dataclass
class DK_ReEigOp(DK_ModOp):
    epsilon: None

    @staticmethod
    def f_of_S(S, epsilon):
        return nn.Threshold(epsilon[0], epsilon[0])(S)

    @staticmethod
    def df_of_S(S, epsilon):
        return S > epsilon[0]


@dataclass
class DK_SqM(DK_ModOp):
    @staticmethod
    def f_of_S(S):
        return torch.sqrt(S)

    @staticmethod
    def df_of_S(S):
        return 0.5 / torch.sqrt(S)


@dataclass
class DK_SqM_inv(DK_ModOp):
    @staticmethod
    def f_of_S(S):
        return 1 / torch.sqrt(S)

    @staticmethod
    def df_of_S(S):
        return -0.5 / torch.sqrt(S) ** 3


class ReEig_I(Function):
    @staticmethod
    def forward(ctx, input, epsilon):
        ctx.save_for_backward(input, epsilon)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s[s < epsilon[0]] = epsilon[0]
            output[k] = u.mm(s.diag().mm(u.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, epsilon = ctx.saved_variables
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1)
            eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(2))
            for k, dx in enumerate(grad_output):
                if len(dx.shape) == 1:
                    continue

                dx = symmetric(dx)

                x = input[k]
                u, s, v = x.svd()

                max_mask = s > epsilon
                s_max_diag = s.clone()
                s_max_diag[~max_mask] = epsilon
                s_max_diag = s_max_diag.diag()
                Q = (
                    max_mask.float().diag()
                )  # for DK this is done in the calculate P function

                dLdV = 2 * (dx.mm(u.mm(s_max_diag)))
                dLdS = eye * (Q.mm(u.t().mm(dx.mm(u))))

                P = calculate_P(s, mode="I")

                grad_input[k] = u.mm(symmetric(P.t() * u.t().mm(dLdV)) + dLdS).mm(u.t())

        return grad_input, None


class ReEig_DK(Function):
    @staticmethod
    def forward(ctx, input, epsilon):
        ctx.save_for_backward(input, epsilon)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s[s < epsilon[0]] = epsilon[0]
            output[k] = u.mm(s.diag().mm(u.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, epsilon = ctx.saved_variables
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1)
            eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(2))
            for k, dx in enumerate(grad_output):
                if len(dx.shape) == 1:
                    continue

                x = input[k]
                u, s, v = x.svd()

                P = calculate_P(s, mode="DK", DK_mod_op=DK_ReEigOp(epsilon=epsilon))

                dLdx = torch.einsum("ji, jk, kl, il, mi, nl -> mn", u, dx, u, P, u, u)
                grad_input[k] = dLdx

        return grad_input, None


def gram_schmidt(V, avoid_zero=True):
    """https://gitlab.lip6.fr/schwander/torchspdnet/-/blob/master/torchspdnet/optimizers.py#L119"""
    n, k = V.shape
    Q = torch.zeros_like(V)
    R = torch.zeros([k, k]).double()

    Q[:, 0] = V[:, 0] / torch.norm(V[:, 0])
    R[0, 0] = Q[:, 0].dot(V[:, 0])

    for i in range(1, k):
        proj = torch.zeros(n).double()
        for j in range(i):
            proj = proj + V[:, i].dot(Q[:, j]) * Q[:, j]
            R[j, i] = Q[:, j].dot(V[:, i])

        if isclose(torch.norm(V[:, i] - proj), torch.DoubleTensor([0])) and avoid_zero:
            Q[:, i] = V[:, i] / torch.norm(V[:, i])
        else:
            Q[:, i] = (V[:, i] - proj) / torch.norm(V[:, i] - proj)

    R[i, i] = Q[:, i].dot(V[:, i])

    return Q, R


def _regularise_with_oas_pytorch(matrix, n_samples, n_features):
    """Recreate regularise with oas func in pytorch"""
    trc = matrix.diagonal(offset=0, dim1=-1, dim2=-2).sum(
        -1
    )  # https://discuss.pytorch.org/t/get-the-trace-for-a-batch-of-matrices/108504
    mu = trc / n_features

    alpha = (
        (matrix**2)
        .view(matrix.shape[0], matrix.shape[1] * matrix.shape[1])
        .mean(axis=1)
    )
    num = alpha + mu**2
    den = (n_samples + 1.0) * (alpha - (mu**2) / n_features)
    shrinkage = torch.clip(num / den, max=1)
    shrunk_cov = (1.0 - shrinkage).view(matrix.shape[0], 1, 1) * matrix
    k = (shrinkage * mu).repeat_interleave(n_features).view(matrix.shape[0], n_features)
    shrunk_cov.diagonal(dim1=-2, dim2=-1)[
        :
    ] += k  # https://discuss.pytorch.org/t/operation-on-diagonals-of-matrix-batch/50779

    return shrunk_cov

def symmetric(A):
    """https://github.com/adavoudi/spdnet/blob/master/spdnet/utils.py"""
    return 0.5 * (A + A.t())


def orthogonal_projection(A, B):
    """https://github.com/adavoudi/spdnet/blob/master/spdnet/utils.py"""
    out = A - B.mm(symmetric(B.transpose(0, 1).mm(A)))
    return out


def retraction(A, ref):
    """https://github.com/adavoudi/spdnet/blob/master/spdnet/utils.py"""
    data = A + ref
    Q, R = data.qr()
    # To avoid (any possible) negative values in the output matrix, we multiply the negative values by -1
    sign = (R.diag().sign() + 0.5).sign().diag()
    out = Q.mm(sign)
    return out

def _get_band_indices(n_chans, n_filters):
    return torch.LongTensor(
        [
            [i for i in range(n_chans * n_filters) if i % n_filters == f]
            for f in range(n_filters)
        ]
    ).flatten()