# PyTorch Indexed Sum Function

In this repo you will find a C implementation of a indexed summation function.

The IndexedSumFunction takes two inputs
1. A 1d array of floats
2. A 1d array of longs that denote increments along input1.

The output is the summation chunked according to the increments.
For example, if input1 is [0.1, 0.2, 0.3, 0.4] and the increments are [3, 1]
then the output will be [0.6, 0.4]
