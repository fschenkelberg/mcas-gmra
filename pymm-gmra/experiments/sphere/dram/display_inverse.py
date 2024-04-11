import os
import sys

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.abspath(os.path.join(_cd_, "..", "..", ".."))]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_

from pysrc.utils.plotting import plot_3d


plot_3d("sphere/results/sphere_inverse.npy", "sphere")