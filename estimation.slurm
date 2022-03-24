#!/bin/sh

#SBATCH -J kpe_est
#SBATCH --time=10-12:00:00
#SBATCH -N 2 -n 2
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 6g

sample=$1
dir=/mnt/pan/Data14/1kgp/${sample}

reads=${dir}/${sample}.mtx
cb=${dir}/${sample}.cb
ase_infer=${dir}/${sample}.ase.infer

python $HOME/ASE/lib/kpe.py ${dir}/${sample} pb   & 
python $HOME/ASE/lib/kpe.py ${dir}/${sample} full &
wait

#pb_est=${dir}/${sample}_pb.est
#full_est=${dir}/${sample}_full.est
#sh $HOME/ASE/split.sh ${pb_est} ${dir}/sub_est/${sample}_pb.est
#sh $HOME/ASE/split.sh ${full_est} ${dir}/sub_est/${sample}_full.est
#
#ase_reform=${dir}/${sample}.ase.reform
#sh $HOME/ASE/split.sh ${ase_reform} ${dir}/sub_ase/${sample}.ase.reform

