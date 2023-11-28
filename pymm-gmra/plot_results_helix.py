# SYSTEM IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import os

# PYTHON PROJECT IMPORTS

def load_data(d: str) -> np.ndarray:
    helix_data: np.ndarray = np.loadtxt(os.path.join(d, "helix.txt"))
    return np.stack([helix_data], axis=0)

def main() -> None:
    cd: str = os.path.abspath(os.path.dirname(__file__))
    helix_dir: str = os.path.join(cd, "experiments", "helix", "results")

    helix_results: np.ndarray = load_data(helix_dir)

    print("loaded data (helix.shape):")
    print(helix_results.shape)

    # features are stored as [data_loading_time,
    #                         covertree_loading_time,
    #                         dyadic_tree_construction_time,
    #                         wavelet construction time
    #                        ]

    # bar chart hyperparameters
    bar_width = 0.35
    opacity = 0.8
    xaxis = np.arange(1)  # Only one dataset

    ###########################################
    # data loading time
    helix_data = helix_results[0, :, 0]
    plt.bar(xaxis,
            np.mean(helix_data, axis=-1),
            bar_width, alpha=opacity, label="helix")
    plt.ylabel("data loading time (s)")
    plt.xticks(xaxis, ["helix"])
    plt.legend()
    plt.show()
    ###########################################

    ###########################################
    # wavelet construction time
    helix_data = helix_results[0, :, 3]
    plt.bar(xaxis,
            np.mean(helix_data, axis=-1),
            bar_width, alpha=opacity, label="helix")
    plt.ylabel("wavelet construction time (s)")
    plt.xticks(xaxis, ["helix"])
    plt.legend()
    plt.show()
    ###########################################

    ###########################################
    # total pymm/data loading time
    helix_data = helix_results[0, :, 3] + helix_results[0, :, 0]
    plt.bar(xaxis,
            np.mean(helix_data, axis=-1),
            bar_width, alpha=opacity, label="helix")
    plt.ylabel("wavelet construction + data loading time (s)")
    plt.xticks(xaxis, ["helix"])
    plt.legend()
    plt.show()
    ###########################################

    ###########################################
    # total script time
    helix_data = helix_results[0, :].sum(axis=-1)
    plt.bar(xaxis,
            np.mean(helix_data, axis=-1),
            bar_width, alpha=opacity, label="helix")
    plt.ylabel("total script time (s)")
    plt.xticks(xaxis, ["helix"])
    plt.legend()
    plt.show()
    ###########################################

if __name__ == "__main__":
    main()
