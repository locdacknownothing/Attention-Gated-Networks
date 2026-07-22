import torch
import torch.utils.data as data
import numpy as np
import datetime
import re
from PIL import Image

from os import listdir
from os.path import join


class DRIVEDataset(data.Dataset):
    def __init__(self, root_dir, split, transform=None, preload_data=False, seed=42, target_size=(592, 592)):
        super(DRIVEDataset, self).__init__()

        self.target_size = target_size

        if split in ['train', 'validation']:
            sub_dir = 'train'
        else:
            sub_dir = 'test'

        image_dir = join(root_dir, sub_dir, 'images')
        label_dir = join(root_dir, sub_dir, 'labels')

        # Collect all image files (.tif)
        all_image_files = sorted([f for f in listdir(image_dir) if f.endswith('.tif')])

        if split in ['train', 'validation']:
            # Group files by base image ID (e.g. 21, 22, etc.) to prevent data leakage
            base_groups = {}
            for img_file in all_image_files:
                match = re.search(r'\d+', img_file)
                base_id = match.group(0) if match else img_file
                if base_id not in base_groups:
                    base_groups[base_id] = []
                base_groups[base_id].append(img_file)

            # Sort base IDs and shuffle with a FIXED random state for consistency across runs
            unique_base_ids = sorted(list(base_groups.keys()))
            rng = np.random.RandomState(seed)
            rng.shuffle(unique_base_ids)

            # 80/20 train/val split on base image groups
            val_count = max(1, int(len(unique_base_ids) * 0.2))
            val_base_ids = set(unique_base_ids[:val_count])
            train_base_ids = set(unique_base_ids[val_count:])

            selected_base_ids = train_base_ids if split == 'train' else val_base_ids
            image_files = []
            for base_id in sorted(list(selected_base_ids)):
                image_files.extend(base_groups[base_id])
            image_files = sorted(image_files)
        else:
            image_files = all_image_files

        self.image_filenames = []
        self.label_filenames = []
        for img_file in image_files:
            if split in ['train', 'validation']:
                name = img_file.replace('_training.tif', '')
            else:
                name = img_file.replace('_test.tif', '')
            label_file = name + '_manual1.png'
            label_path = join(label_dir, label_file)
            self.image_filenames.append(join(image_dir, img_file))
            self.label_filenames.append(label_path)

        assert len(self.image_filenames) == len(self.label_filenames)

        # Data augmentation
        self.transform = transform

        # Preload into RAM
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = []
            self.raw_labels = []
            for img_f, lbl_f in zip(self.image_filenames, self.label_filenames):
                img_pil = Image.open(img_f).convert('RGB').resize(self.target_size, Image.BILINEAR)
                lbl_pil = Image.open(lbl_f).convert('L').resize(self.target_size, Image.NEAREST)
                
                img_t = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() # (3, H, W)
                lbl_t = torch.from_numpy((np.array(lbl_pil) > 127).astype(np.float32)).unsqueeze(0) # (1, H, W)
                
                self.raw_images.append(img_t)
                self.raw_labels.append(lbl_t)
            print('Loading is done\n')

        # Report
        print('Number of {0} images: {1}'.format(split, self.__len__()))

    def __getitem__(self, index):
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        if not self.preload_data:
            img_pil = Image.open(self.image_filenames[index]).convert('RGB').resize(self.target_size, Image.BILINEAR)
            lbl_pil = Image.open(self.label_filenames[index]).convert('L').resize(self.target_size, Image.NEAREST)

            input = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float()
            target = torch.from_numpy((np.array(lbl_pil) > 127).astype(np.float32)).unsqueeze(0)
        else:
            input = self.raw_images[index].clone()
            target = self.raw_labels[index].clone()

        if self.transform:
            input, target = self.transform(input, target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
