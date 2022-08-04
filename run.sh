#!/bin/bash

echo "Starting server"
/usr/bin/python3 server.py &
sleep 5 # Sleep for 3s to give the server enough time to start

for i in `seq 0 9`; do
    echo "Starting client $i"
    /usr/bin/python3 client.py &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait