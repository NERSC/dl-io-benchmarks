import time
from dataloaders.rbc_h5_multifiles_dataloader import get_data_loader
import argparse

def main(args):
  dataset = get_data_loader(args.files_pattern,
                            batch_size=args.batch_size,
                            num_workers=args.num_data_workers,
                            crop_size=args.crop_size) 
  total_time = 0
  for epoch in range(args.epochs+1):
    print("epoch", epoch)
    t_start = time.time()
    for idx, data in enumerate(dataset):
      if idx > args.max_batches_per_epoch:
        break
      pass

    if epoch > 0:
      total_time += time.time() - t_start

  n_batches = min(args.max_batches_per_epoch+1, len(dataset))
  print("Timing:", float(args.batch_size*n_batches*args.epochs)/(total_time), "samples/s")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--files_pattern", default='', type=str)
  parser.add_argument("--batch_size", default=64, type=int)
  parser.add_argument("--crop_size", default=256, type=int,
                      help="crop_size of crop from larger input image")
  parser.add_argument("--epochs", default=2, type=int)
  parser.add_argument("--max_batches_per_epoch", default=100, type=int)
  parser.add_argument("--num_data_workers", default=4, type=int)
  args = parser.parse_args()
  main(args)
