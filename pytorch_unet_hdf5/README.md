## pytorch-dataloader
Benchmark data loading only, no network training.

## Installation
Create a Python 3 conda environment (either with the anaconda installer or
through `spack install anaconda3@2020.07`) in `INSTALL_DIR` and install the
following packages:

```
$ INSTALL_DIR=/path/to/pytorch_dataloader

$ conda create -y --prefix $INSTALL_DIR python=3.8 \
    ninja mkl mkl-include numpy pyyaml setuptools cmake cffi typing \
    h5py ipython ipykernel matplotlib scikit-learn pandas pillow \
    pytorch
```

## Input files creation

The files to use as input can be generated randomly, since the process
doesn't perform any actual operation on the data, but just moves bytes.

```python
import h5py
import numpy as np

n_samples = 100
with h5py.File("data.h5", "w") as hf:
    hf.create_dataset('fields_tilde_upsampled', data=np.random.normal(size=(n_samples, 4, 512, 512)))
    hf.create_dataset('fields_hr', data=np.random.normal(size=(n_samples, 4, 512, 512)))
```

You can tweak the `n_samples` variable, which is in linear correlation with
the size of the output file; at `100` it creates a 1.6 GB file, at `1000` it
creates a 16 GB file, at 5000 a 80 GB file, etc.

## Usage 

Load/activate the conda environment and you should be good to go:
```
$ conda activate $INSTALL_DIR
$ python benchmark_dataloader.py \
    --files_pattern "/path/to/input/file.h5" \
    --batch_size 64 \
    --crop_size 256 \
    --num_data_workers 4
```

### Usage on Cori
To submit a cgpu test, use [submit_cgpu.sh](submit_cgpu.sh). The script will
test three cases: reading directly from Lustre scratch, reading from cgpu nvme
nodes, reading from a shifter perNodeCache (XFS file).

You can use [submit_haswell.sh](scripts/submit_haswell.sh) to test reading from
Lustre from a Haswell node.

The latest benchmarks and the logs of runs you perform on Cori can be found in
the [logs](logs) directory.
