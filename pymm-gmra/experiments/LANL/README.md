
Additive GMRA (AGMRA) Proof of Concept

This directory contains the proof of concept implementation for Additive GMRA (AGMRA). In this implementation, each JSON file is computed with Iterative Processing of Blocked node embeddings.

Explanation

The process of computing blocks is designed to tackle the challenges associated with processing large datasets. The dataset is partitioned into blocks, ensuring each block encapsulates a manageable subset of data points. This partitioning strategy facilitates iterative processing, where each block undergoes independent computation. This modular approach not only enhances computational efficiency but also enables the potential for parallel processing, a significant advantage in handling large datasets.

Within the processing loop for each block, a series of operations are executed. This includes the computation of essential data structures such as the: covertrees, dyadic trees, and wavelet trees. The low-dimensional embeddings are then derived from these structures, providing a condensed representation of the dataset while preserving essential features.

A critical aspect of the implementation is the meticulous handling of the last block, which may contain a smaller number of data points than the computed block size. This consideration ensures that all data points are processed accurately, mitigating any potential discrepancies in the final analysis.
