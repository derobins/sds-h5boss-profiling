Profiling code for h5boss-like I/O

The test program creates a number of datasets and then fills them with
(garbage) data. It is intended to mimic h5boss I/O patterns.

Dataset creation and dataset writes are monitored and reported separately.

The number and sizes of the datasets are given via the input file. A
sample input file (dset_stats_100.csv) is provided.

To build, simply compile the .c file with h5pcc and run it with mpiexec.
Note that you'll need to have librados installed in the normal place.

    h5pcc -o h5boss_io_pattern h5boss_io_pattern.c -lrados
