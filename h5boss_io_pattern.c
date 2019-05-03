/* h5boss_io_pattern.c
 *
 * This is a performance testing program that simulates h5boss I/O
 * patterns.
 *
 *
 * NOTE:
 *
 * This will need some fixin' after we move to the independent
 * RADOS VOL connector (as opposed to the one that is built with
 * the HDF5 library that uses private functions). In particular,
 * the new H5VLrados_init() call takes different parameters.
 */

#include <ctype.h>              /* getopt processing */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <unistd.h>             /* getopt processing */

#include <rados/librados.h>     /* RADOS */

#include <mpi.h>

#include <hdf5.h>
#include <H5VLrados_public.h>   /* RADOS VOL connector */

#include "timer.h"              /* Local timer functions */


/* ERROR MACROS */
#define ERROR_MSG                   {fprintf(stderr, "***ERROR*** at line %d\n", __LINE__);}
#define GOTO_ERROR                  {ERROR_MSG goto error;}
/* This one is only for use in getopt processing */
#define GOTO_ERROR_USAGE(s)         {if (0 == rank) {ERROR_MSG usage(stderr); fprintf(stderr, "%s\n", (s));} goto error;}

/* The maximum number of datasets we'll read in */
#define MAX_DATASETS    25304802

/* Max # of characters in the dataset name */
#define DSET_NAME_MAX 48

/* Timer IDs */
#define DSET_CREATE_TIMER   0
#define DSET_WRITE_TIMER    1

/* Timer globals */
struct timeval start_time[3];
float elapse[3];

static void
usage(FILE * stream) {
    fprintf(stream, "\n");
    fprintf(stream, "Usage: mpiexec -n <# proc> ./h5boss_io_pattern -i <input_filename> -o <output_hdf5_filename>\n");
    fprintf(stream, "       [-d <n_max_datasets>] [-v <rados|native>] [-c <rados_config_file>] [-p <rados_pool_name>]\n");
    fprintf(stream, "\n");
    fprintf(stream, "\t-c <rados_config_file>       RADOS config file name. Must be specified if -v rados set.\n");
    fprintf(stream, "\n");
    fprintf(stream, "\t-d <n_max_datasets>          (optional) Max number of dataset lines to process in the input file.\n");
    fprintf(stream, "\n");
    fprintf(stream, "\t-g                           (optional) Turn dataset names like /name/of/dset into a series of nested groups.\n");
    fprintf(stream, "\t                             Default is to not do this, munge '/' to '-' in the name, and create a single dataset.\n");
    fprintf(stream, "\n");
    fprintf(stream, "\t-h                           Print this help message.\n");
    fprintf(stream, "\n");
    fprintf(stream, "\t-i <input_filename>          Input dataset list filename.\n");
    fprintf(stream, "\n");
    fprintf(stream, "\t-o <output_hdf5_filename>    Output HDF5 filename.\n");
    fprintf(stream, "\n");
    fprintf(stream, "\t-p <rados_pool_name>         RADOS pool name. Must be specified if -v rados set.\n");
    fprintf(stream, "\n");
    fprintf(stream, "\t-v <rados|native>            (optional) Whether to use the native or RADSOS VOL connector.\n");
    fprintf(stream, "\t                             Default is the native VOL connector.\n");
    fprintf(stream, "\n");
}

static int
setup_rados_vol(hid_t fapl_id, const char *rados_config_file, const char *rados_pool) {
    rados_t cluster;        /* The RADOS cluster */

    /* Set up RADOS for testing */
    if(rados_create(&cluster, NULL) < 0)
        GOTO_ERROR
    if(rados_conf_read_file(cluster, rados_config_file) < 0)
        GOTO_ERROR

    /* Initialize the VOL connector */
    if(H5VLrados_init(cluster, rados_pool) < 0)
        GOTO_ERROR

    /* Set up the fapl to use the RADOS-API VOL connector */
    if(H5Pset_fapl_rados(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL) < 0)
        GOTO_ERROR
    if(H5Pset_all_coll_metadata_ops(fapl_id, true) < 0)
        GOTO_ERROR

    printf("RADOS-API VOL connector set up correctly\n");

    return 0;

error:
    printf("***ERROR*** RADOS-API VOL connector NOT set up correctly!\n");
    return -1;
} /* end setup_rados_vol() */

static int
read_input_file(const char *filename, int munge_dataset_names, int max_datasets, /*out*/ int *actual_datasets, 
        /*out*/ char **dset_info_names, /*out*/ int *dset_info_sizes, /*out*/ long long *max_size, 
        /*out*/ long long *total_size)
{
    FILE    *f = NULL;              /* input file pointer */
    char    line_buf[256];          /* buffer for reading file lines */
    char    dset_ndim[4];           /* buffer for storing dataset dimension string */
    int     dset_nelem;             /* number of dataset elements */
    int     i, j;                   /* iterators */

    printf("Reading from %s\n", filename);

    /* Open the file */
    if (NULL == (f = fopen(filename, "r")))
        GOTO_ERROR

    /* Read lines from the file until we hit the end or the
     * maximum number of desired datasets.
     */
    i = 0;
    *max_size = 0;
    *total_size = 0;
    while ((i < max_datasets) && (fgets(line_buf, 255, f) != NULL)) {

        /* Scan the line from the file. Lines look like this:
         *
         * /3523/55144/1/coadd, 1D: 4619, 147808
         *
         * dataset name, # dimensions: number of elements, wtf
         */
        sscanf(line_buf + 1, "%[^,], %[^:]: %d, %d", dset_info_names[i], dset_ndim, &dset_nelem, &dset_info_sizes[i]);

        /* Keep track of the largest size and total size */
        if (dset_info_sizes[i] > *max_size)
            *max_size = dset_info_sizes[i];
        *total_size += dset_info_sizes[i];

        /* All the test datasets should be 1D. Warn if not. */
        if (strcmp(dset_ndim, "1D") != 0) {
            printf("Dimension is %s!\n", dset_ndim);
            continue;
        }
        
        /* Since the dataset names look like paths, we have to
         * substitute the '/' character or HDF5 will get grumpy.
         */
        if (munge_dataset_names)
            for (j = 0; j < strlen(dset_info_names[i]); j++) {
                if (dset_info_names[i][j] == '/') 
                    dset_info_names[i][j] = '-';
            }

        i++;
    }

    /* Close the input file */
    fclose(f);

    /* The actual number of datasets, based on the file */
    *actual_datasets = i;

    printf("Number of datasets read: %d\n", *actual_datasets);

    return 0;

error:
    return -1;
} /* end read_input_file() */

int
main(int argc, char* argv[])
{
    /* General things */
    char *in_filename = NULL;               /* input filename */
    char *out_filename = NULL;              /* output filename */
    int max_datasets = MAX_DATASETS;        /* maximum number of datasets to create */
    int n_datasets = -1;                    /* actual number of datasets to create */
    long long max_size = -1;                /* maximum dataset size */
    long long total_size = -1;              /* total size (bytes) of all datasets */
    char *io_buf = NULL;                    /* buffer for dataset writes */
    int c;                                  /* getopt option */
    int i;                                  /* iterator */
    long long lli;                          /* iterator */

    /* MPI things */
    int size = -1;                          /* MPI size */
    int rank = -1;                          /* MPI rank */
    int my_n_datasets = -1;                 /* # of datasets this rank will create */
    int my_start_dataset = -1;              /* in the array of dataset info, where this rank will start */

    /* RADOS things */
    int use_rados = 0;                      /* whether or not to use RADOS */
    char *rados_config_file = NULL;         /* name of the RADOS config file */
    char *rados_pool = NULL;                /* name of the RADOS pool */

    /* HDF5 things */
    int create_intermediate_groups = 0;     /* whether to create intermediate groups or just a single dataset */
    hid_t fapl_id = H5I_INVALID_HID;        /* file access property list ID */
    hid_t fid = H5I_INVALID_HID;            /* file ID */
    hid_t fsid = H5I_INVALID_HID;           /* file dataspace ID */
    hid_t lcpl_id = H5I_INVALID_HID;        /* link creation property list ID */
    hid_t did = H5I_INVALID_HID;            /* dataset ID */
    hsize_t dims[3] = {0};                  /* dataspace dimensions */

    /* h5boss dataset info things */
    char **dset_info_names = NULL;          /* pointers into the 1D string array (for fake 2D string array) */
    char *dset_info_names_1d = NULL;        /* 1D array of (concatenated) dataset names */
    int *dset_info_sizes = NULL;            /* the size of each dataset */


    /* Set up MPI (early so only rank 0 emits issues on command-line errors) */
    if (MPI_Init(&argc, &argv) < 0)
        GOTO_ERROR
    if (MPI_Comm_size(MPI_COMM_WORLD, &size) < 0)
        GOTO_ERROR
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) < 0)
        GOTO_ERROR

    /*********************/
    /* COMMAND LINE ARGS */
    /*********************/

    opterr = 0;

    while ((c = getopt (argc, argv, "c:d:ghi:o:p:v:")) != -1)
        switch (c)
        {
            case 'c':
                rados_config_file = optarg;
                break;
            case 'd':
                max_datasets = atoi(optarg);
                break;
            case 'g':
                create_intermediate_groups = 1;
                break;
            case 'h':
                if (0 == rank)
                    usage(stdout);
                return EXIT_SUCCESS;
                break;
            case 'i':
                in_filename = optarg;
                break;
            case 'o':
                out_filename = optarg;
                break;
            case 'p':
                rados_pool = optarg;
                break;
            case 'v':
                if (!strcmp(optarg, "rados"))
                    use_rados = 1;
                else if (strcmp(optarg, "native")) {
                    if (0 == rank)
                        fprintf(stderr, "Unknown VOL connector %s passed via -v\n", optarg);
                    GOTO_ERROR_USAGE("");
                }
                break;
            case '?':
                if (optopt == 'c' || optopt == 'd' || optopt == 'i' || optopt == 'o' || optopt == 'p' || optopt == 'v')
                    if (0 == rank)
                        fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint(optopt))
                    if (0 == rank)
                        fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    if (0 == rank)
                        fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
                GOTO_ERROR_USAGE("");
            default:
                GOTO_ERROR_USAGE("How did we get here?")
        }

    /* Sanity checks on command line */
    if (argc < 3)
        GOTO_ERROR_USAGE("Not enough command-line arguments");

    /* Input and output files must be specified */
    if (!in_filename)
        GOTO_ERROR_USAGE("Input file must be specified on the command line with -i")
    if (!out_filename)
        GOTO_ERROR_USAGE("Output file must be specified on the command line with -o")

    /* if RADOS was specified, we have to have the pool and config files set */
    if (use_rados) {
        if (!rados_pool)
            GOTO_ERROR_USAGE("If you want RADOS, you have to specify the pool name with -p")
        if (!rados_config_file)
            GOTO_ERROR_USAGE("If you want RADOS, you have to specify the config file name with -c")
    }

    /***************************/
    /* SETUP AND FILE CREATION */
    /***************************/

    /* Set up the RADOS VOL connector */
    if (H5I_INVALID_HID == (fapl_id = H5Pcreate(H5P_FILE_ACCESS)))
        GOTO_ERROR
    if (H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL) < 0)
        GOTO_ERROR
    if (use_rados)
        if (setup_rados_vol(fapl_id, rados_config_file, rados_pool) < 0)
            GOTO_ERROR

    /* Set up 'dataset info' memory
     *
     * The two char arrays are so we can fake a 2D string array with two allocations.
     */
    if (NULL == (dset_info_names = (char **)calloc(max_datasets, sizeof(char *))))
        GOTO_ERROR
    if (NULL == (dset_info_names_1d = (char *)calloc(max_datasets * DSET_NAME_MAX, sizeof(char))))
        GOTO_ERROR
    if (NULL == (dset_info_sizes = (int *)calloc(max_datasets, sizeof(int))))
        GOTO_ERROR
    /* Pointer fixup for the fake 2D string array */
    for (i = 0; i < max_datasets; i++) 
        dset_info_names[i] = dset_info_names_1d + i * DSET_NAME_MAX;

    /* Rank 0 reads the input file */
    if (0 == rank) {
        int munge_dataset_names = !create_intermediate_groups;

        if (read_input_file(in_filename, munge_dataset_names, max_datasets, &n_datasets, dset_info_names, dset_info_sizes, &max_size, &total_size) < 0)
            GOTO_ERROR
    }

    /* Broadcast the input file data */
    MPI_Bcast(&n_datasets, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_size, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(dset_info_names_1d, n_datasets * DSET_NAME_MAX, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(dset_info_sizes, n_datasets, MPI_INT, 0, MPI_COMM_WORLD);

    /* Determine this rank's subset
     * Note that the last rank may get some extra work
     */
    my_n_datasets = n_datasets / size;
    my_start_dataset = rank * my_n_datasets;
    if (rank == size - 1) 
        my_n_datasets += n_datasets % size;

    /********************/
    /* DATASET CREATION */
    /********************/

    /* Create the HDF5 file */
    if (H5I_INVALID_HID == (fid = H5Fcreate(out_filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id)))
        GOTO_ERROR

    /* Create the link access property list
     *
     * As an aside, this code was originally intended to interpret the
     * dataset names as a group hierarchy which should be created by
     * this code. The NERSC tests did not do this, however, and just
     * munged the filename. Using -g optionally restores this original
     * intent.
     */
    if (H5I_INVALID_HID == (lcpl_id = H5Pcreate(H5P_LINK_CREATE)))
        GOTO_ERROR
    if (create_intermediate_groups)
        if (H5Pset_create_intermediate_group(lcpl_id, 1) < 0)
            GOTO_ERROR

    /* Start timing dataset creation */
    MPI_Barrier(MPI_COMM_WORLD);
    timer_on(DSET_CREATE_TIMER); 

    /* Create all the datasets */
    for (i = 0; i < n_datasets; i++) {

        /* Create dataspace */
        dims[0] = dset_info_sizes[i];
        if (H5I_INVALID_HID == (fsid = H5Screate_simple(1, dims, NULL)))
            GOTO_ERROR

        /* Create dataset */
        if (H5I_INVALID_HID == (did = H5Dcreate(fid, dset_info_names[i], H5T_NATIVE_CHAR, fsid, lcpl_id, H5P_DEFAULT, H5P_DEFAULT)))
            GOTO_ERROR

        /* Close up */
        if (H5Sclose(fsid) < 0)
            GOTO_ERROR
        if (H5Dclose(did) < 0)
            GOTO_ERROR
    }

    /* Stop timing dataset creation */
    MPI_Barrier(MPI_COMM_WORLD);
    timer_off(DSET_CREATE_TIMER);

    /* Close the file */
    if (H5Fclose(fid) < 0)
        GOTO_ERROR

    /*****************/
    /* DATASET WRITE */
    /*****************/

    /* Open the file */
    if (H5I_INVALID_HID == (fid = H5Fopen(out_filename, H5F_ACC_RDWR, fapl_id)))
        GOTO_ERROR

    /* Allocate a buffer for I/O and fill it with random data */
    if (NULL == (io_buf = (char *)calloc((size_t)max_size, sizeof(char))))
        GOTO_ERROR
    srand(time(NULL));
    for(lli = 0; lli < max_size; lli++)
        io_buf[lli] = (char)(rand() % 255);

    /* Start timing dataset writes */
    MPI_Barrier(MPI_COMM_WORLD);
    timer_on(DSET_WRITE_TIMER); 

    /* Write junk data to the datasets */
    for (i = 0; i < my_n_datasets; i++) {

        /* Open the dataset */
        if (H5I_INVALID_HID == (did = H5Dopen(fid, dset_info_names[my_start_dataset + i], H5P_DEFAULT)))
            GOTO_ERROR

        /* Write the data */
        if (H5Dwrite(did, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, io_buf) < 0)
            GOTO_ERROR

        /* Close the dataset */
        H5Dclose(did);
    }

    /* Stop timing dataset writes */
    MPI_Barrier(MPI_COMM_WORLD);
    timer_off(DSET_WRITE_TIMER);

    /* Close the file */
    if (H5Fclose(fid) < 0)
        GOTO_ERROR

    /* Free the buffer */
    free(io_buf);

    /***********/
    /* CLEANUP */
    /***********/

    /* Rank 0 dumps the timing info */
    if (rank == 0) {
        /* Dataset creation cost */
        printf("Dataset creation time (s): ");
        timer_msg(DSET_CREATE_TIMER);

        /* Dataset write cost */
        printf("Dataset write time (s): ");
        timer_msg(DSET_WRITE_TIMER);

        /* Total size */
        printf("Total of dataset data sizes: ");
        printf("%lld MB\n", total_size / 1024 / 1024);
    }

    /* memory */
    free(dset_info_names);
    free(dset_info_names_1d);
    free(dset_info_sizes);

    /* HDF5 */
    if (H5Pclose(fapl_id) < 0)
        GOTO_ERROR

    /* MPI */
    if (MPI_Finalize() < 0)
        GOTO_ERROR

    return EXIT_SUCCESS;

error:
    /* Badness - clean up as best we can */

    /* memory */
    free(dset_info_names);
    free(dset_info_names_1d);
    free(dset_info_sizes);
    free(io_buf);

    /* HDF5 */
    H5E_BEGIN_TRY {
        H5Fclose(fid);
        H5Pclose(fapl_id);
        H5Pclose(lcpl_id);
        H5Sclose(fsid);
        H5Dclose(did);
    } H5E_END_TRY;

    /* MPI */
    MPI_Finalize();

    return EXIT_FAILURE;
} /* end main() */

