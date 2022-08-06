import os
from torch.utils.data import Dataset, DataLoader


def construct_dataloder(dataset, batch_size, shuffle=True, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)


class Dataset(Dataset):
    def __init__(self, file_ids, labels, data_dir):
        self.file_ids = file_ids
        self.labels = labels
        self.data_dir = data_dir

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        image_id = self.file_ids[index]
        x = torch.load(os.path.join(self.data_dir, image_id))
        y = self.labels[id]
        x = self.augment(x)
        return x, y

    def augment(self, x):
        """Augmentations for images"""
        return x
