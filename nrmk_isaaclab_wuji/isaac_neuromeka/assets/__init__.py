# Modified from Orbit's Assets

import os

import toml

# Conveniences to other module directories via relative paths
ORBIT_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ORBIT_ASSETS_METADATA = toml.load(os.path.join(ORBIT_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ORBIT_ASSETS_METADATA["package"]["version"]


##
# Configuration for different assets.
##

from .dual_arm import *
from .indy import *
from .moby import *
from .moby200 import *
from .nami import *
from .wuji import *
from .zen import *
