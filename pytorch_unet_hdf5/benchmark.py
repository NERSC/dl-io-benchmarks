import os
import time
from dataloaders.rbc_h5_multifiles_dataloader import get_data_loader
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from models.UNet import UNet

def main(args):
  dataset = get_data_loader(args.files_pattern,
                            batch_size=args.batch_size,
                            num_workers=args.num_data_workers,
                            crop_size=args.crop_size if args.crop_size != 0 else None)

  device = torch.cuda.current_device()
  if args.dataloader_only and dist.get_rank() == 0:
    print("Running in dataloader_only mode ...")
  else:
    model = UNet().to(device)
    if dist.is_initialized():
      model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=[args.local_rank])

    if args.forward_only:
      if dist.get_rank() == 0:
        print("Running in inference (forward only) mode ...")
    else:
      if dist.get_rank() == 0:
        print("Running in training (forward/backward) mode ...")
      optimizer = torch.optim.Adam(model.parameters())

  total_time = 0
  for epoch in range(args.epochs+1):
    if dist.get_rank() == 0:
      print("epoch", epoch)
    t_start = time.time()
    for idx, data in enumerate(dataset):
      if idx > args.max_batches_per_epoch:
        break

      if args.dataloader_only:
        continue

      inp, tar = map(lambda x: x.to(device), data)

      if args.forward_only:
        model.eval()
        gen = model(inp)
      else:
        model.zero_grad()
        model.train()
        gen = model(inp)
        loss = torch.nn.functional.l1_loss(gen, tar)
        loss.backward()
        optimizer.step()

    if epoch > 0:
      total_time += time.time() - t_start

  n_batches = min(args.max_batches_per_epoch+1, len(dataset))
  if dist.get_rank() == 0:
    print("Timing:", float(args.batch_size*n_batches*args.epochs)/(total_time), "samples/s")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--files_pattern", default='', type=str)
  parser.add_argument("--batch_size", default=64, type=int)
  parser.add_argument("--crop_size", default=0, type=int,
                      help="crop_size of crop from larger input image. Default (0) means no cropping.")
  parser.add_argument("--epochs", default=2, type=int)
  parser.add_argument("--max_batches_per_epoch", default=100, type=int)
  parser.add_argument("--num_data_workers", default=4, type=int)
  parser.add_argument("--dataloader_only", action='store_true', default=False)
  parser.add_argument("--forward_only", action='store_true', default=False, help="Only works if dataloader_only is False")
  parser.add_argument("--local_rank", default=0, type=int)
  args = parser.parse_args()

  if 'WORLD_SIZE' in os.environ:
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

  if dist.get_rank() == 0:
    print(args)

  main(args)
