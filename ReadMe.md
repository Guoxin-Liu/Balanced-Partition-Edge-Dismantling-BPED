# Balanced-Partition Edge Dismantling

This repository contains a standalone Python implementation of BPED and an example using the US power-grid network.

## Run the example

Use Python 3.11 or newer. From the repository directory, run

```shell
python -m pip install -r requirements.txt
python bped.py
```

The demo uses the bundled edge list, sets the dismantling threshold to 0.01, and uses random seed 0. It prints the metrics and saves a [dismantling curve PDF](output/pdf/power-US-Grid_bped.pdf) under the repository's `output/pdf/` directory. The PDF shows one run.

To change the input, threshold, seed, or output path, run

```shell
python bped.py power-US-Grid.edgelist --gamma 0.01 --seed 7 --output power_seed7.pdf
```

The default input and output paths are relative to the script. Paths passed on the command line are relative to the current directory unless absolute. Custom edge lists use two whitespace-separated integer node IDs per edge. Lines starting with `#` are comments. The demo treats edges as undirected, discards self-loops and duplicates, and uses the largest connected component.

## How BPED works

1. **Build connected blocks.** Use the dismantling threshold γ to set a common upper bound of ⌊γN⌋ nodes per block. Grow blocks from uniformly random unassigned seeds using local boundary costs. Merge adjacent blocks where the size limit permits, refine the partition with local node moves, then split any disconnected blocks and merge feasible neighbors again.
2. **Compress the graph.** Represent each block as a vertex whose mass is its node count. The weight between two quotient vertices counts the original edges linking their blocks.
3. **Order the removals.** Recursively bisect the largest remaining quotient component by original node count. Each split uses a mass-weighted Fiedler ordering and a RatioCut sweep. Remove the original edges crossing each split in the resulting order.

## Reading the curve

Let N and M be the node and edge counts of the input component. After each edge removal, the curve records the size of the largest connected component divided by N. The dashed line marks the threshold γ, and the dot marks the first point at or below it.

- **q_c** is the fraction of edges removed when the largest component first reaches or falls below the size limit.
- **Threshold R** sums the relative largest-component sizes after each deletion through that first crossing and divides by M.
- **Full R** sums the same quantity through all M deletions and divides by M. After the threshold, the demo continues by spectrally bisecting the largest remaining component on its original nodes and removing all edges crossing the split before choosing the next component.

## Data

The bundled Power edge list comes from the [Network Repository](https://networkrepository.com/power-US-Grid.php).
