# GMRA Algorithm Example

## Introduction
This code is an example that demonstrates how to extract low-dimensional embeddings for points in a dataset using the Geometric Multi-Scale Resolution Analysis (GMRA) algorithm. The GMRA algorithm is inspired by the behavior of the cortex and processes a point cloud at different scales to produce increasingly fine-grained manifolds.

## Background Information
In the context of the GMRA algorithm, low-dimensional features are stored within the wavelet nodes themselves. To extract these features for specific points in the dataset at a known scale, you must traverse the tree to the depth for that scale and then inspect the nodes to get the features.

At the correct depth in the GMRA tree, a single point will reside within one of the nodes at that particular level. To locate a single point, you can either follow the tree's hierarchical structure downward, employing a Depth-First Search (DFS) approach, or you can traverse down to the desired depth and systematically examine each node individually.

If the goal is to extract features for more than a single point, you'll need to aggregate information across multiple nodes at the same depth. It's important to note that the dimensionality of the features stored in one node at a specific depth may differ from the dimensionality in another node at the same depth. This variation in dimensionality reflects the adaptability of the GMRA algorithm to capture intricate data structures and provide meaningful embeddings at various scales.

## Code Explanation
Here's a simple explanation of the code:

First, the necessary libraries and modules are imported. Then the sphere function generates a simple sphere shape dataset consisting of 3D points.

The main function, get_embeddings, is defined to extract low-dimensional embeddings at a specified scale from the GMRA tree.

The main function is defined to load data, construct a cover tree, a dyadic tree, and a wavelet tree, and then extract the low-dimensional embeddings.

The get_embeddings function is called with the wavelet tree, the desired scale, and the dataset to retrieve the low-dimensional embeddings for the points at the specified scale.

Finally, the obtained embeddings are saved to a text file.

# INSTRUCTIONS
Please make sure that you've built and installed the c++ code to whatever python environment
you're going to use. This can be done from the pymm-gmra directory by typing:
```<python_interpreter> setup.py install```

## Running the Code
To run this code, you should follow these steps:

Navigate to the directory where the code is located:
```bash
cd ./scratch/f006dg0/mcas-gmra/pymm-gmra/experiments.
```

Execute the first Python script to build the cover tree:
```bash
python ./sphere/covertree_build.py ./sphere/results
```

Please know that this script takes one required argument and one optional argument. The required argument is the path to a directory. If that directory doesn't exist, the code will attempt to create it for you. In this directory is where the build covertree will serialize to (as a json file). The optional argument is whether or not to validate the tree after construction (expensive operation).

Then execute the second Python script to run the GMRA algorithm and extract low-dimensional embeddings:
```bash
python ./sphere/dram/gmra.py ./sphere/results/sphere_covertree.json
```

After running these commands, the low-dimensional embeddings will be extracted and saved to the "helix/results" directory.