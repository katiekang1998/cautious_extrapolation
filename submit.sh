#!/bin/bash

mkdir logs/out/ -p
mkdir logs/err/ -p

sbatch --array=1-1%1 sbatch.sh