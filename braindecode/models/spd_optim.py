from torch import nn

from .spd_functional import gram_schmidt, orthogonal_projection, retraction


class StiefelParameter(nn.Parameter):
    """A kind of Variable that is to be considered a module parameter on the space of
    Stiefel manifold.
    """

    def __new__(cls, data=None, requires_grad=True):
        return super(StiefelParameter, cls).__new__(
            cls, data, requires_grad=requires_grad
        )

    def __repr__(self):
        return "Parameter containing:" + self.data.__repr__()


class AltStiefelMetaOptimizer(
    object
):  # adapted for toggling the weird 0 set of the weights
    """This is a meta optimizer which uses other optimizers for updating parameters
    and remap all StiefelParameter parameters to Stiefel space after they have been updated.

    reset_zero: True: retract with correct term, False: Retract with correct term plus old weight (?)

    For proper implementation want set_zero=False, reset_zero=True
    """

    def __init__(
        self,
        optimizer,
        reset_zero=True,
        use_gram_schmidt=False,
        use_alt_proj=False,
    ):
        self.optimizer = optimizer
        self.state = {}
        self.reset_zero = reset_zero
        self.use_gram_schmidt = use_gram_schmidt
        self.use_alt_proj = use_alt_proj

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, *args, **kwargs):
        return self.optimizer.load_state_dict(*args, **kwargs)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if isinstance(p, StiefelParameter):

                    # store stiefel weights pre-update for retraction post-update
                    if id(p) not in self.state:
                        self.state[id(p)] = p.data.clone()
                    else:
                        self.state[id(p)].fill_(0).add_(p.data)

                    # egrad to rgrad
                    if self.use_alt_proj:
                        rgrad = proj_tanX_stiefel(
                            p.grad.data.unsqueeze(0).unsqueeze(0),
                            p.data.unsqueeze(0).unsqueeze(0),
                        ).squeeze()
                    else:
                        rgrad = orthogonal_projection(p.grad.data, p.data)

                    # replace egrad with rgrad
                    p.grad.data.fill_(0).add_(rgrad)

                    # need to set to 0 so p.data contains just the update term
                    # to be subtracted from the weight
                    if self.reset_zero:
                        p.data.fill_(0)

        loss = self.optimizer.step(closure)

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):

                    # keep the weights on the stiefel manifold
                    if self.use_gram_schmidt:
                        trans = gram_schmidt(p.data + self.state[id(p)])[0]
                    else:
                        trans = retraction(p.data, self.state[id(p)])

                    p.data.fill_(0).add_(trans)

        return loss


def proj_tanX_stiefel(x, X):
    """
    https://gitlab.lip6.fr/schwander/torchspdnet/-/blob/master/torchspdnet/optimizers.py
    Projection of x in the Stiefel manifold's tangent space at X"""
    return x - X.matmul(x.transpose(2, 3)).matmul(X)