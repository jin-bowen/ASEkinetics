#!/bin/sh

#SBATCH -J kpe_est
#SBATCH -p gpu -C gpuk40 --gres=gpu:1
#SBATCH -N 1 -n 8
#SBATCH --mem=32g
#SBATCH --time=4-12:00:00

ind=$1
dir=/mnt/pan/Data14/1kgp

#head -1 ${dir}/${ind}/sub_parser/${ind}_1.ase > ${dir}/${ind}/${ind}.ase
#for file in `ls ${dir}/${ind}/sub_parser/${ind}_*.ase`
#do sed 1d $file  >> ${dir}/${ind}/${ind}.asedone
#done

reads=${dir}/${ind}/${ind}.mtx
ase=${dir}/${ind}/${ind}.ase
cb=${dir}/${ind}/${ind}.cb
ase_infer=${dir}/${ind}/${ind}.ase.infer

#python $HOME/ASE/lib/ase_infer.py ${reads} ${ase} ${cb} ${ind} ${dir}/${ind}
#python $HOME/ASE/lib/kpe_cellcycle.py ${ase_infer} ${ind}_cellQC.tsv ${ind}_cellcyle.mle 

python $HOME/ASE/lib/kpe.py ${ase_infer} full  ${ind}_full.est  &
python $HOME/ASE/lib/kpe.py ${ase_infer} pb    ${ind}_pb.est    &
python $HOME/ASE/lib/kpe.py ${ase_infer} gibbs ${ind}_gibbs.est &
wait

