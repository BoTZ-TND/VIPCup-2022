import os
import yaml
import glob
import numpy as np
import shutil
import pandas as pd

import random_operations


class MakeDataset:
    def __init__(self, config_fp: str):
        with open(config_fp, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.data_root_dir = self.config['dataset_root_dir']
        self.synth_dirs = self.config['synthetic_dirs']
        self.real_dirs = self.config['real_dirs']
        self.sampling_config = self.config['sampling_configs']
        self.rng = np.random.default_rng(seed=self.sampling_config['seed'])

    @staticmethod
    def check_img(file_name: str):
        return file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif',
                                           '.tiff', '.bmp', '.gif'))

    def get_file_names(self, folder):
        files = glob.glob(os.path.join(folder, '**'), recursive=True)
        files = [f for f in files if self.check_img(f)]
        return files

    @staticmethod
    def copy_files(files: list, out_dir: str, out_ids: list):
        for f, out_id in zip(files, out_ids):
            shutil.copy(f, os.path.join(out_dir, out_id))

    def split_files(self, image_ids):
        n_train = int(
            len(image_ids) * self.sampling_config['train_ratio'] + 0.5)
        n_val = int(len(image_ids) * self.sampling_config['val_ratio'] + 0.5)
        n_test = int(len(image_ids) * self.sampling_config['test_ratio'] + 0.5)
        self.rng.shuffle(image_ids)
        train_ids = image_ids[:n_train]
        val_ids = image_ids[n_train:n_train + n_val]
        test_ids = image_ids[n_train + n_val:]
        return train_ids, val_ids, test_ids

    def make_dataset(self):
        syn_images = []
        real_images = []
        for name, syn_dir in self.synth_dirs.items():
            t_images = self.get_file_names(self.data_root_dir + syn_dir['path'])
            n = int(0.5 + self.sampling_config['output_size'] * \
                    self.sampling_config['synthetic_ratio'] * syn_dir['ratio'])
            if n > len(t_images):
                n = len(t_images)
            t_images = self.rng.choice(t_images, n, replace=False)
            syn_images.extend(t_images)
        for name, real_dir in self.real_dirs.items():
            t_images = self.get_file_names(self.data_root_dir + real_dir['path'])
            n = int(0.5 + self.sampling_config['output_size'] * \
                    (1 - self.sampling_config['synthetic_ratio']) * real_dir[
                        'ratio'])
            if n > len(t_images):
                n = len(t_images)
            t_images = self.rng.choice(t_images, n, replace=False)
            real_images.extend(t_images)

        # Copy images to data folder
        data_out_dir = os.path.join(self.config['data_out_dir'],
                                    str(self.config['version']))
        if os.path.isdir(data_out_dir):
            shutil.rmtree(data_out_dir)
        image_dir = os.path.join(data_out_dir, 'images')
        os.makedirs(image_dir)
        syn_ids = [f'syn_{i}{os.path.splitext(f)[1]}' for i, f in enumerate(syn_images)]
        real_ids = [f'real_{i}{os.path.splitext(f)[1]}' for i, f in enumerate(real_images)]
        self.copy_files(syn_images, image_dir, syn_ids)
        self.copy_files(real_images, image_dir, real_ids)

        # Random Crop and Augmentation
        random_operations.random_operations(image_dir, os.path.join(image_dir, '..', 'augmented'),
                                            self.sampling_config['seed'])

        # make labels
        syn_labels = [1 for _ in syn_images]
        real_labels = [0 for _ in real_images]
        image_ids = syn_ids + real_ids
        image_paths = syn_images + real_images
        labels = syn_labels + real_labels
        label_file = os.path.join(data_out_dir, 'labels.csv')
        df = pd.DataFrame({'image_ids': image_ids, 'label': labels,
                           'image_path': image_paths})
        df.to_csv(label_file, index=False)
        print('Saved labels to', label_file)

        # make train, val, test splits
        s_train, s_val, s_test = self.split_files(syn_ids)
        r_train, r_val, r_test = self.split_files(real_ids)

        train_ids = s_train + r_train
        val_ids = s_val + r_val
        test_ids = s_test + r_test

        train_file = os.path.join(data_out_dir, 'train.txt')
        val_file = os.path.join(data_out_dir, 'val.txt')
        test_file = os.path.join(data_out_dir, 'test.txt')
        with open(train_file, 'w') as f:
            f.write('\n'.join(train_ids))
        with open(val_file, 'w') as f:
            f.write('\n'.join(val_ids))
        with open(test_file, 'w') as f:
            f.write('\n'.join(test_ids))
        print('Saved train, val, test splits to', train_file, val_file,
              test_file)

        print('images:', len(image_ids))
        print('syn_images:', len(syn_images))
        print('real_images:', len(real_images))
        print('train:', len(train_ids))
        print('val:', len(val_ids))
        print('test:', len(test_ids))

        print('Done!')


if __name__ == '__main__':
    make_dataset = MakeDataset('src/data/dataset_constructor.yaml')
    make_dataset.make_dataset()
