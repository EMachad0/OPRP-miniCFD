#!/bin/bash

# temporarily cd to script folder
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

mpirun --machinefile hosts.txt -np 8 miniCFD.out

