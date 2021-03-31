#!/bin/bash -l
#SBATCH --time=01:30:00
#SBATCH -C haswell
##SBATCH --account=m1759
#SBATCH -q regular
#SBATCH --nodes=1

#SBATCH --output logs/dataloader-haswell-%j.out
#SBATCH --image=nersc/pytorch:ngc-20.07-v0
#SBATCH --volume="/global/cscratch1/sd/mustafa/tmpfiles:/tmp_xfs:perNodeCache=size=200G"

module load pytorch/1.7.1
data_file="/global/cscratch1/sd/jpathak/rbc2d/dataSR/dedalus/upsampled/raw/chunks/testing_pairs_1.h5"

printf "****** Benchmarking reading from scratch ******\n"
python benchmark.py --files_pattern $data_file \
                    --dataloader_only \
                    --batch_size 64 \
                    --crop_size 256 \
                    --num_data_workers 4


printf "\n"
printf "****** Benchmarking reading from xfs ******\n"
srun --nodes=1 --ntasks=1 shifter --env HDF5_USE_FILE_LOCKING=FALSE <<EOF
cp $data_file /tmp_xfs/file.h5
python benchmark.py --files_pattern /tmp_xfs/file.h5 \
                    --dataloader_only \
                    --batch_size 64 \
                    --crop_size 256 \
                    --num_data_workers 4
EOF
