import os
import shutil
import math
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str,default='/data1/huangwenmin/CelebAMask-HQ/CelebA-HQ-img')
parser.add_argument('--label_path', type=str, default='/data1/huangwenmin/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt')
parser.add_argument("--target_path", type=str,default='./databese')
parser.add_argument("--mode", type=str, default='val')
parser.add_argument("--start", type=int, default=3002)
parser.add_argument("--end", type=int, default=30002)
opts = parser.parse_args()

target_path = opts.target_path

os.makedirs(target_path, exist_ok=True)

Tags_Attributes = {
    'Bangs': ['with', 'without'],
    'Eyeglasses': ['with', 'without'],
    'HairColor': ['black', 'blond', 'brown'],
}

if opts.mode=='train':
    for tag in Tags_Attributes.keys():
        for attribute in Tags_Attributes[tag]:
            open(os.path.join(target_path, f'{tag}_{attribute}.txt'), 'w')

if opts.mode=='val':
    for tag in Tags_Attributes.keys():
        for attribute in Tags_Attributes[tag]:
            open(os.path.join(target_path, f'{tag}_{attribute}_val.txt'), 'w')

# celeba-hq
celeba_imgs = opts.img_path
celeba_label = opts.label_path

with open(celeba_label) as f:
    lines = f.readlines()
if opts.mode=='val':
    opts.start=2
    opts.end=3002
for line in tqdm.tqdm(lines[opts.start:opts.end]):

    line = line.split()

    filename = os.path.join(os.path.abspath(celeba_imgs), line[0])

    if opts.mode=='train':
    # Use only gender and age as tag-irrelevant conditions. Add other labels if you want.
        if int(line[6]) == 1:
            with open(os.path.join(target_path, 'Bangs_with.txt'), mode='a') as f:
                f.write(f'{filename} {line[21]} {line[40]}\n')
        elif int(line[6]) == -1:
            with open(os.path.join(target_path, 'Bangs_without.txt'), mode='a') as f:
                f.write(f'{filename} {line[21]} {line[40]}\n')

        if  int(line[16]) == 1:
            with open(os.path.join(target_path, 'Eyeglasses_with.txt'), mode='a') as f:
                f.write(f'{filename} {line[21]} {line[40]}\n')
        elif int(line[16]) == -1:
            with open(os.path.join(target_path, 'Eyeglasses_without.txt'), mode='a') as f:
                f.write(f'{filename} {line[21]} {line[40]}\n')

        if int(line[9]) == 1 and int(line[10]) == -1 and int(line[12]) == -1 and int(line[18]) == -1:
            with open(os.path.join(target_path, 'HairColor_black.txt'), mode='a') as f:
                f.write(f'{filename} {line[21]} {line[40]}\n')
        elif int(line[9]) == -1 and int(line[10]) == 1 and int(line[12]) == -1 and int(line[18]) == -1:
            with open(os.path.join(target_path, 'HairColor_blond.txt'), mode='a') as f:
                f.write(f'{filename} {line[21]} {line[40]}\n')
        elif int(line[9]) == -1 and int(line[10]) == -1 and int(line[12]) == 1 and int(line[18]) == -1:
            with open(os.path.join(target_path, 'HairColor_brown.txt'), mode='a') as f:
                f.write(f'{filename} {line[21]} {line[40]}\n')
    if opts.mode=='val':
        if int(line[6]) == 1:
            with open(os.path.join(target_path, 'Bangs_with_val.txt'), mode='a') as f:
                f.write(f'{filename} {line[21]} {line[40]}\n')
        elif int(line[6]) == -1:
            with open(os.path.join(target_path, 'Bangs_without_val.txt'), mode='a') as f:
                f.write(f'{filename} {line[21]} {line[40]}\n')

        if int(line[16]) == 1:
            with open(os.path.join(target_path, 'Eyeglasses_with_val.txt'), mode='a') as f:
                f.write(f'{filename} {line[21]} {line[40]}\n')
        elif int(line[16]) == -1:
            with open(os.path.join(target_path, 'Eyeglasses_without_val.txt'), mode='a') as f:
                f.write(f'{filename} {line[21]} {line[40]}\n')

        if int(line[9]) == 1 and int(line[10]) == -1 and int(line[12]) == -1 and int(line[18]) == -1:
            with open(os.path.join(target_path, 'HairColor_black_val.txt'), mode='a') as f:
                f.write(f'{filename} {line[21]} {line[40]}\n')
        elif int(line[9]) == -1 and int(line[10]) == 1 and int(line[12]) == -1 and int(line[18]) == -1:
            with open(os.path.join(target_path, 'HairColor_blond_val.txt'), mode='a') as f:
                f.write(f'{filename} {line[21]} {line[40]}\n')
        elif int(line[9]) == -1 and int(line[10]) == -1 and int(line[12]) == 1 and int(line[18]) == -1:
            with open(os.path.join(target_path, 'HairColor_brown_val.txt'), mode='a') as f:
                f.write(f'{filename} {line[21]} {line[40]}\n')
    


