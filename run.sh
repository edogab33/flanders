#!/bin/bash

for i in `seq 0 6`; do
    echo "Starting experiment # $i"
    python main.py --seed=123 --exp_num=$i;
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait