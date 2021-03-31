import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch import Tensor
import h5py

def get_data_loader(files_pattern, batch_size, num_workers, crop_size=None):
  dataset = GetDataset(files_pattern, crop_size)
  dataloader = DataLoader(dataset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=torch.cuda.is_available())

  return dataloader

class GetDataset(Dataset):
  def __init__(self, files_pattern, crop_size):
    self.files_pattern = files_pattern
    self.inp_field = 'fields_tilde_upsampled'
    self.tar_field = 'fields_hr'
    self.crop_size = crop_size
    self._get_files_stats()

  def _get_files_stats(self):
    self.files_paths = glob.glob(self.files_pattern)
    self.n_files = len(self.files_paths)
    with h5py.File(self.files_paths[0], 'r') as _f:
      self.n_samples_per_file = _f[self.inp_field].shape[0]
      self.inp_size = _f[self.inp_field].shape[2]
      self.tar_size = _f[self.tar_field].shape[2]
    self.n_samples = self.n_files * self.n_samples_per_file

    if self.crop_size and (self.inp_size != self.tar_size):
      if dist.get_rank() == 0:
        print("ERROR: Cropping is not implemented if input and target are not the same size. Aborting ...")
      exit()

    self.files = [None for _ in range(self.n_files)]
    if dist.get_rank() == 0:
      print("Found {} at path {}. Number of examples: {}".format(self.n_files, self.files_pattern, self.n_samples))

  def _open_file(self, ifile):
    self.files[ifile] = h5py.File(self.files_paths[ifile], 'r')

  def __len__(self):
    return self.n_samples

  def __getitem__(self, global_idx):
    ifile = int(global_idx/self.n_samples_per_file)
    local_idx = int(global_idx%self.n_samples_per_file)

    if not self.files[ifile]:
      self._open_file(ifile)

    if self.crop_size:
      rnd_x = random.randint(0, self.inp_size-self.crop_size)
      rnd_y = random.randint(0, self.inp_size-self.crop_size)
      return torch.as_tensor(self.files[ifile][self.inp_field][local_idx][:, rnd_x:rnd_x+self.crop_size, rnd_y:rnd_y+self.crop_size]), \
             torch.as_tensor(self.files[ifile][self.tar_field][local_idx][:, rnd_x:rnd_x+self.crop_size, rnd_y:rnd_y+self.crop_size])
    else:
      return torch.as_tensor(self.files[ifile][self.inp_field][local_idx]), \
             torch.as_tensor(self.files[ifile][self.tar_field][local_idx])
