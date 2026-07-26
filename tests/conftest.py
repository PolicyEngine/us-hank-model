import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run slow full-grid tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow full-grid test; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def ss():
    """Production-grid two-asset steady state, solved once per session."""
    from us_hank import model
    return model.solve_steady_state()


@pytest.fixture(scope="session")
def G_jac(ss):
    from us_hank import model
    return model.solve_jacobian(ss, T=300)
