#!/bin/bash

# This script is used to rerun the program when it inevitably segfaults
exit_code=1
while [ $exit_code -ne 0 ]; do
    "$@"
    exit_code=$?
    sleep 1
done
