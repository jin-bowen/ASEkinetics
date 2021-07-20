#!/bin/sh

#SBATCH -J kpe_eval
#SBATCH -p gpu -C gpuk40 --gres=gpu:1
#SBATCH -N 1 -n 8
#SBATCH --mem=32g
#SBATCH --time=4-12:00:00

kpe=$1
method=$2
dir=/mnt/pan/Data14/1kgp

prefix=${kpe%.est}
python $HOME/ASE/lib/evaluation.py ${dir}/${kpe} ${method} ${dir}/${prefix}_${method}.eval 

