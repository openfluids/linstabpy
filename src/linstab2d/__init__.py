import sys
import warnings

warnings.warn(
    "Importing from linstabpy is deprecated. Please import from linstabpy instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Alias linstabpy to linstabpy
import linstabpy
sys.modules["linstabpy"] = linstabpy
