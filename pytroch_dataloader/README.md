### pytorch-dataloader
Benchmark data loading only, no network training.

### Usage:
To submit a cgpu test:

```bash
sbatch scripts/submit_cgpu.sh
```
The script will test three cases: reading directly from Lustre scratch, reading from cgpu nvme nodes, reading from a shifter perNodeCache (XFS file).

You can use `scripts/submit_haswell.sh` to test reading from Lustre from a Haswell node.