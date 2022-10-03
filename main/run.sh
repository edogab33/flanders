#!/bin/bash

for i in `seq 0 1`; do
    echo "Starting experiment # $i"
    python cifar_server.py --num_rounds=1 --exp_num=$i;
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait