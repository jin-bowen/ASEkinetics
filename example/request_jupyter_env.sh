#!/bin/sh

srun --x11 -c 2 --mem=8gb --pty bash
jupyter notebook --ip=127.0.0.1 &
module unload base
firefox &

