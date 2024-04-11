import os
import sys

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.abspath(os.path.join(_cd_, "..", "..", ".."))]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_

from pysrc.utils.plotting import *

if __name__ == "__main__":
	plot_3d("helix/results/helix_inverse.npy", "helix")