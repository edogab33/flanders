#!/bin/bash

for i in `seq 0 17`; do
    echo "Starting experiment # $i"
    python main.py --seed=123 --exp_num=$i;
done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait