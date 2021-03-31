#!/bin/bash -l
#SBATCH --time=1:20:00
#SBATCH -C gpu
#SBATCH --account=m1759
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10

#SBATCH --output logs/unet-train-cgpu-%j.out
#SBATCH --image=nersc/pytorch:ngc-20.07-v0
#SBATCH --volume="/global/cscratch1/sd/mustafa/tmpfiles:/tmp_xfs:perNodeCache=size=200G"

module load pytorch/1.7.1-gpu

set -f
data_file="/global/cscratch1/sd/jpathak/rbc2d/dataSR/dedalus/upsampled/raw/chunks/testing_pairs_1.h5"

printf "****** Benchmarking reading from scratch ******\n"
# python -m torch.distributed.launch --nproc_per_node=8 \
python benchmark.py --files_pattern $data_file \
                    --batch_size 64 \
                    --crop_size 256 \
                    --num_data_workers 4



printf "\n"
printf "****** Benchmarking reading from nvme ******\n"
printf "Copying data file to nvme..."
set +f
time cp $data_file /tmp/

set -f
# python -m torch.distributed.launch --nproc_per_node=8 \
python benchmark.py --files_pattern /tmp/* \
                    --batch_size 64 \
                    --crop_size 256 \
                    --num_data_workers 4
