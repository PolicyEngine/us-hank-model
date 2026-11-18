"""The package version and the installed distribution metadata agree."""
from importlib.metadata import version

import us_hank


def test_version_matches_distribution_metadata():
    assert version("us-hank-model") == us_hank.__version__
