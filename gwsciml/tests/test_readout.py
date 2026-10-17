import numpy as np
import torch

from gwsciml.models.readout import LikelihoodMapReadout, MonotonePL
from gwsciml.pe.prior import log_lambda


def test_init_equals_em_statistic():
    rng = np.random.default_rng(0)
    L = rng.normal(5, 3, (7, 4, 3)).astype(np.float32)
    lp, lt = np.log(rng.dirichlet(np.ones(4))), np.log(rng.dirichlet(np.ones(3)))
    m = LikelihoodMapReadout(lp, lt)
    np.testing.assert_allclose(m.score(torch.as_tensor(L)).detach().numpy(), log_lambda(L, lp, lt), rtol=1e-5)


def test_monotone_g():
    g = MonotonePL([1.0, 3.0, 5.0])
    with torch.no_grad():
        g.dslope.copy_(torch.tensor([-3.0, 0.5, 2.0]))      # would give a negative slope without the clamp
    x = torch.linspace(-2, 10, 500)
    assert torch.all(torch.diff(g(x)) > 0)
