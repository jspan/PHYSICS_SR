import os
import argparse

import skimage
import skimage.io as sio

import torch

parser = argparse.ArgumentParser(description='Pre-processing DIV2K .jpeg images')

parser.add_argument('--pathFrom', default='../../../../dataset/DIV2K',
                    help='directory of images to convert')
parser.add_argument('--pathTo', default='../../../../dataset/DIV2K_decoded',
                    help='directory of images to save')
parser.add_argument('--split', default=False,
                    help='save individual images')
parser.add_argument('--select', default='',
                    help='select certain path')

args = parser.parse_args()

for (path, dirs, files) in os.walk(args.pathFrom):
    print(path)
    targetDir = path.replace(args.pathFrom, args.pathTo)
    if len(args.select) > 0 and path.find(args.select) == -1:
        continue

    if not os.path.exists(targetDir):
        os.mkdir(targetDir)

    if len(dirs) == 0:
        pack = {}
        n = 0
        for fileName in files:
            (idx, ext) = os.path.splitext(fileName)
            if ext == '.jpeg':
                png = sio.imread(os.path.join(path, fileName))
                tensor = torch.Tensor(png.astype(float)).byte()
                if args.split:
                    torch.save(tensor, os.path.join(targetDir, idx + '.pt'))
                else:
                    pack[int(idx.split('x')[0])] = tensor
                n += 1
                if n % 100 == 0:
                    print('Converted ' + str(n) + ' images.')
        if len(pack) > 0:
            torch.save(pack, targetDir + '/pack.pt')
            print('Saved pt binary.')
            del pack
