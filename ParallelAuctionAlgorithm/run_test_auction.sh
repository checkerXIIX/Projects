make
#test_file num_runs output_filename
ser_out=$(./test_auction_serial 50 auc_ser)
par_omp_out=$(./test_auction_parallel_openmp 50 auc_par_omp_4)
par_man_out=$(./test_auction_parallel_man 50 auc_par_man_4)

parse_ser=($ser_out)
parse_par_omp=($par_omp_out)
parse_par_man=($par_man_out)
