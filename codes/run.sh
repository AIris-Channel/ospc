cat local_test/test_stdin/stdin.csv | deepspeed --include localhost:0 main.py \
    1>local_test/test_output/stdout.csv \
    2>local_test/test_output/stderr.csv
