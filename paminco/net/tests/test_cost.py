import pytest
import numpy as np

from itertools import zip_longest

from paminco.net import load_sioux
from paminco.net._data_gas import temporary_gas_files
from paminco.net._data_examples import (NET_SIMPLE_POLYNOMIAL,
                                       NET_ELECTRICAL_PIECEWISE)
from paminco.net.network import Network
from paminco.net.cost import EquidistantInterpolationRule, PiecewiseQuadraticCost, SymbolicCost, SimplePolynomial, BreakpointsInterpolationRule, EdgeCostInterpolation
from paminco.algo.mca import MCAInterpolationRule


def test_costfunc_traffic():
    net = load_sioux()
    fft = net.cost.coefficients[:, 0]
    a = net.cost.coefficients[:, 4]
    coeffs = {"a": a, "fft": fft}
    F = "x*fft + a/5 * x**5"
    f = "fft + a*x**4"
    f1 = "4*a*x**3"
    f2 = "12*a*x**2"
    sc = SymbolicCost(coeffs, F, f, f1, f2, shared=net.shared)
    net.integrate_cost()
    for d in range(4):
        for i in range(5):
            x = np.random.random(net.m) * 100
            polyval = net.cost(x, d=d)
            symval = sc(x, d=d)
            assert np.allclose(polyval, symval)
        for i in range(5):
            x = np.random.random(net.m) * 1000
            polyval = net.cost(x, d=d)
            symval = sc(x, d=d)
            assert np.allclose(polyval, symval)
        for i in range(5):
            x = np.random.random(net.m) * 10000
            polyval = net.cost(x, d=d)
            symval = sc(x, d=d)
            assert np.allclose(polyval, symval)


def test_costfunc_gas():
    # Open Gas40 network
    with temporary_gas_files("gas40") as tmpfiles:
        gas40 = Network.from_gaslib(*tmpfiles)
    F = "beta * x * abs(x)"
    f = "2 * beta * abs(x)"
    f1 = "2 * beta * x / abs(x)"
    f2 = "0"
    beta = gas40.cost.coefficients[:, 2]
    sc = SymbolicCost({"beta": beta}, F, f, f1, f2, shared=gas40.shared)
    for d in range(4):
        for _ in range(5):
            x = np.random.random(gas40.m) * 100
            polyval = gas40.cost(x, d=d)
            symval = sc(x, d=d)
            assert np.allclose(polyval, symval)
        for _ in range(5):
            x = np.random.random(gas40.m) * 1000
            polyval = gas40.cost(x, d=d)
            symval = sc(x, d=d)
            assert np.allclose(polyval, symval)
        for _ in range(5):
            x = np.random.random(gas40.m) * 10000
            polyval = gas40.cost(x, d=d)
            symval = sc(x, d=d)
            assert np.allclose(polyval, symval)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_derivation_integration(d):
    net = load_sioux()
    c2 = net.cost.integrate(d=d)
    c3 = c2.derivate(d=d)
    f = np.random.random(net.m) * 10000
    assert np.allclose(c3(f), net.cost(f))
    assert np.allclose(c3(f, d=1), net.cost(f, d=1))
    
    # TODO-PW: gas networks


def test_polycost_exact():
    """
    Tests the SimplePolynomial and PolynomialCost classes against
    precalculated values of the polynomials (1+x)^i, i = 1, ..., 10
    for x = -5, ..., 5
    """
    target_vals = np.array([[-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
                            [16, 9, 4, 1, 0, 1, 4, 9, 16, 25, 36],
                            [-64, -27, -8, -1, 0, 1, 8, 27, 64, 125, 216],
                            [256, 81, 16, 1, 0, 1, 16, 81, 256, 625, 1296],
                            [-1024, -243, -32, -1, 0, 1, 32, 243, 1024, 3125, 7776],
                            [4096, 729, 64, 1, 0, 1, 64, 729, 4096, 15625, 46656],
                            [-16384, -2187, -128, -1, 0, 1, 128, 2187, 16384, 78125, 279936],
                            [65536, 6561, 256, 1, 0, 1, 256, 6561, 65536, 390625, 1679616],
                            [-262144, -19683, -512, -1, 0, 1, 512, 19683, 262144, 1953125, 10077696],
                            [1048576, 59049, 1024, 1, 0, 1, 1024, 59049, 1048576, 9765625, 60466176]]).T

    coeff_raw_data = [[1, 1],
                      [1, 2, 1],
                      [1, 3, 3, 1],
                      [1, 4, 6, 4, 1],
                      [1, 5, 10, 10, 5, 1],
                      [1, 6, 15, 20, 15, 6, 1],
                      [1, 7, 21, 35, 35, 21, 7, 1],
                      [1, 8, 28, 56, 70, 56, 28, 8, 1],
                      [1, 9, 36, 84, 126, 126, 84, 36, 9, 1],
                      [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]]

    coefficients = np.array(list(zip_longest(*coeff_raw_data, fillvalue=0))).T
    signed = [False for _ in range(len(coefficients))]

    m = len(coefficients)
    m_edges = [[str(e), str(e + 1)] for e in range(m)]

    dummy_net = Network(m_edges, cost_data=(coefficients, signed))
    polynomials = [SimplePolynomial(coefficients[i], signed[i]) for i in range(len(coefficients))]

    for i, x in enumerate(range(-5, 6)):
        poly_result = np.array([p(x) for p in polynomials])
        assert np.array_equal(poly_result, dummy_net.cost(x))
        assert np.array_equal(poly_result, target_vals[i])


def test_signed_polycost():
    """Test signed polycost/simple polynomials against precalculated values."""
    target_vals = np.array([[335, 169, 71, 23, 7, 5, 11, 39, 107, 233, 435],
                            [315, 156, 63, 18, 3, 0, 9, 42, 117, 252, 465],
                            [-315, -156, -63, -18, -3, 0, 9, 42, 117, 252, 465]]).T

    coefficients = np.array([[5, 1, 2, 3], [0, 3, 3, 3], [0, 3, 3, 3]])
    signed = [True, True, False]

    m = len(coefficients)
    m_edges = [[str(e), str(e + 1)] for e in range(m)]

    dummy_net = Network(m_edges, cost_data=(coefficients, signed))
    polynomials = [SimplePolynomial(coefficients[i], signed[i]) for i in range(len(coefficients))]

    for i, x in enumerate(range(-5, 6)):
        poly_result = np.array([p(x) for p in polynomials])
        assert np.array_equal(poly_result, dummy_net.cost(x))
        assert np.array_equal(poly_result, target_vals[i])


def numerical_function_compare(f1, f2, a, b, k, exact=False, x_shape=None):
    """Compare two functions f1 and f2 by checking values numerically.
    
    Inserts ``k`` values between ``a`` and ``b`` (including both) into the
    functions ``f1`` and ``f2``.
    If ``exact`` is set to True, the values must match exacly, otherwise
    numpy.isclose is used.  If ``x_shape`` is set to something else than
    None, instead of a real value x from [a, b] a numpy array of shape
    ``x_shape`` with constant value x is inserted into f1 and f2.
    """
    step = (b - a) / (k - 1)
    xs = np.arange(a, b + step, step)
    for x in xs:
        if x_shape is not None:
            x = np.full(x_shape, x)
            if exact:
                if not np.array_equal(f1(x), f2(x)):
                    return False
            else:
                if not np.allclose(f1(x), f2(x)):
                    return False
        else:
            if exact:
                if not f1(x) == f2(x):
                    return False
            else:
                if not np.isclose(f1(x), f2(x)):
                    return False
    return True


def test_piecewise_quadratic_cost():
    net = Network.from_xml(NET_ELECTRICAL_PIECEWISE)

    F = [lambda x: 0.5 * x * x if x < 3 else 2.5 * x * x - 12 * x + 18,
         lambda x: 0.5 * x * x if x < 2 else 1.5 * x * x - 4 * x + 4,
         lambda x: 0.5 * x * x if x < 1 else 2 * x * x - 3 * x + 1.5]

    f = [lambda x: x if x < 3 else 5 * x - 12,
         lambda x: x if x < 2 else 3 * x - 4,
         lambda x: x if x < 1 else 4 * x - 3]

    def target_cost(x): return np.array([F[i](x[i]) for i in range(net.m)])
    def target_der(x): return np.array([f[i](x[i]) for i in range(net.m)])

    for i in range(net.m):
        assert numerical_function_compare(F[i], net.cost[i], 0, 10, 1000)
        def der(x): return net.cost[i](x, d=1)
        assert numerical_function_compare(f[i], der, 0, 10, 1000)

    assert numerical_function_compare(target_cost, net.cost, 0, 10, 1000, x_shape=net.m)
    def der(x): return net.cost(x, d=1)
    assert numerical_function_compare(target_der, der, 0, 10, 1000, x_shape=net.m)


@pytest.mark.parametrize(
    "coefficients, smoothness",
    [
        (
            np.array([[-np.inf, -np.inf, np.inf, np.inf],
                      [4, 3, 1, 0],
                      [2, 7, -1, 1],
                      [np.inf, np.inf, np.inf, 42]]),
            [True, True, False]
        ),
        (
            np.array([[4, 3, 1, 0],
                      [2, 7, -1, 1]]),
            [True, True, False]
        ),
        (
            np.array([[4, 3, 1, 0],
                      [2, 7, -1, 1],
                      [3, 1, 7, 2]]),
            [True, False, False]
        ),
        (
            np.array([[4, 3, 1, 0],
                      [2, 7, -1, 1],
                      [3, 1, 42, 2]]),
            [False, False, False]
        ),
        (
            np.array([[-np.inf, -np.inf, np.inf, -np.inf],
                      [2, 2, 2, -1],
                      [2, 2, 2, 1],
                      [np.inf, np.inf, np.inf, 8]]),
            [True, True, True]
        )
    ]
)
def test_piecewise_quadratic_single_edge_smoothness(coefficients, smoothness, margin=0):
    ei = np.full(len(coefficients), 0)
    pwq = PiecewiseQuadraticCost((coefficients, ei))
    for k, s in enumerate(smoothness):
        assert pwq.is_smooth(k=k, margin=margin) == s


def test_breakpoints_interpolation():
    net = Network.from_xml(NET_SIMPLE_POLYNOMIAL)

    x_max = 5000
    bps = np.unique(np.random.randint(-x_max + 1, x_max, 100))

    bp_targets = []

    # Create target breakpoints for all edges
    # By selecting the appropriate breakpoints from bp
    # and adding possible artificial breakpoints
    for e in range(net.m):
        l, u = net.edges.lb[e], net.edges.ub[e]
        pre = np.array([-np.inf, max(l, -x_max)])
        selected_bps = bps[(bps > l) & (bps < u)]
        post = np.array([]) if u == np.inf else np.array([u])
        bp_targets.append(np.concatenate([pre, selected_bps, post]))
        
    irule = BreakpointsInterpolationRule(bps)
    pwqc = net.cost.interpolate(irule, x_max=x_max)

    # Get the breakpoints from the interpolated cost
    df = pwqc._ec.to_df()
    for i in range(len(bp_targets)):
        bp = np.array((df[df["edge"] == i]["tau"]))
        assert len(bp) == len(bp_targets[i])
        assert np.array_equal(bp, np.array(bp_targets[i]))


def test_equidistant_interpolation():
    net = Network.from_xml(NET_SIMPLE_POLYNOMIAL)

    x_max = 5000
    delta = 10

    irule = EquidistantInterpolationRule(delta)
    pwqc = net.cost.interpolate(irule, x_max=x_max)
    df = pwqc._ec.to_df()

    # Check breakpoints by testing if they equal arange
    for i in range(net.m):
        bp = np.array((df[df["edge"] == i]["tau"]))
        lo, up = net.edges.lb[i], net.edges.ub[i]
        lo = max(lo, -x_max)
        up = min(up, x_max)
        tbp = np.arange(lo, up + 1, delta)
        tbp = np.concatenate([np.array([-np.inf]), tbp])
        assert np.array_equal(bp, tbp)


def test_mca_interpolation_rule():
    net = Network.from_xml(NET_SIMPLE_POLYNOMIAL)

    bp_targets = [[-np.inf, 0, 2, 8, 26, 80, 242, 728, 2186, 6560, 19682,
                   59048, 177146, 531440, 1594322, 4782968, 14348906],
                  [-np.inf, 0, 2, 6.47213578e+00,
                   1.95700045e+01, 5.87610787e+01, 1.76300253e+02, 5.28906430e+02,
                   1.58672118e+03, 4.76016418e+03, 1.42804927e+04, 4.28414783e+04,
                   1.28524435e+05, 3.85573305e+05, 1.15671991e+06, 3.47015974e+06,
                   1.04104792e+07, 3.12314377e+07],
                  [-np.inf, -1.43489060e+07, -4.78296867e+06, -1.59432289e+06,
                   -5.31440963e+05, -1.77146988e+05, -5.90489959e+04, -1.96829986e+04,
                   -6.56099949e+03, -2.18699968e+03, -7.28999435e+02, -2.42998440e+02,
                   -8.09953646e+01, -2.69861071e+01, -8.95827470e+00, -2.87339972e+00,
                   -5.70660096e-01, 0.00000000e+00, 2.00000000e+00, 6.47213578e+00,
                   1.95700045e+01, 5.87610787e+01, 1.76300253e+02, 5.28906430e+02,
                   1.58672118e+03, 4.76016418e+03, 1.42804927e+04, 4.28414783e+04,
                   1.28524435e+05, 3.85573305e+05, 1.15671991e+06, 3.47015974e+06,
                   1.04104792e+07, 3.12314377e+07],
                  [-np.inf, -1000, -3.33332333e+02, -1.11107778e+02,
                   -3.70269251e+01, -1.23152862e+01, -4.02349000e+00, -1.07989912e+00,
                   0.00000000e+00, 2.00000000e+00, 6.47213578e+00, 1.95700045e+01,
                   5.87610787e+01, 1.76300253e+02, 5.28906430e+02, 1.58672118e+03,
                   4.76016418e+03, 1.42804927e+04, 4.28414783e+04, 1.28524435e+05,
                   3.85573305e+05, 1.15671991e+06, 3.47015974e+06, 1.04104792e+07,
                   3.12314377e+07],
                  [-np.inf, -1.43489060e+07, -4.78296867e+06, -1.59432289e+06,
                   -5.31440963e+05, -1.77146988e+05, -5.90489959e+04, -1.96829986e+04,
                   -6.56099949e+03, -2.18699968e+03, -7.28999435e+02, -2.42998440e+02,
                   -8.09953646e+01, -2.69861071e+01, -8.95827470e+00, -2.87339972e+00,
                   -5.70660096e-01, 0.00000000e+00, 2.00000000e+00, 6.47213578e+00,
                   1.95700045e+01, 5.87610787e+01, 1.76300253e+02, 5.28906430e+02,
                   1000],
                  [-np.inf, -1000, -3.33332333e+02, -1.11107778e+02,
                   -3.70269251e+01, -1.23152862e+01, -4.02349000e+00, -1.07989912e+00,
                   0.00000000e+00, 2.00000000e+00, 6.47213578e+00,
                   1.95700045e+01, 5.87610787e+01, 1.76300253e+02, 5.28906430e+02,
                   1000]]
    exact = [0]

    x_max = 14348906

    rule = MCAInterpolationRule(2, 3 * net.m * x_max, net.m, x_max)
    pwqc = net.cost.interpolate(rule)
    # Get the breakpoints from the interpolated cost
    df = pwqc._ec.to_df()
    for i in range(len(bp_targets)):
        bp = np.array((df[df["edge"] == i]["tau"]))
        assert len(bp) == len(bp_targets[i])
        if i in exact:
            assert np.array_equal(bp, np.array(bp_targets[i]))
        else:
            assert np.allclose(bp, np.array(bp_targets[i]))


@pytest.mark.parametrize("rng", [42, 4242, 424242, 0, 123])
def test_edge_interpolation(rng):
    """Test the computation of the computation of piecewise quadratic pieces.
    
    Creates random polynomials and interpolates them with equidistant steps.
    Checks consistency by testing that
    a) the derivative of the cost function coincides with the linear interpolation
       2 * a * x + b  at the breakpoints
    b) the linear interpolation is continuous

    Does not check the actual function but only the interpolation of
    the derivative. In particuar no test of the offsets.
    """
    rng = np.random.default_rng(rng)
    degree = rng.integers(3, 6)
    coeffs = rng.integers(-10, 10, degree + 1)
    ec = SimplePolynomial(coeffs)
    step = 10
    irule = EquidistantInterpolationRule(step)
    eci = EdgeCostInterpolation(0, ec, irule, 1000 + step, 0, 1000 + step)
    coeff = eci.interpolate()
    a = 2 * coeff[1:-1, 0]
    b = coeff[1:-1, 1]
    tau = coeff[1:-1, 3]
    for i, t in enumerate(tau):
        assert ec.ddx(t) == a[i] * t + b[i]
        if i > 0:
            assert a[i] * t + b[i] == a[i - 1] * t + b[i - 1]
