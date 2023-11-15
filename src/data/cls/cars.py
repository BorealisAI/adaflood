import pathlib
import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image

from torchvision import datasets
from torchvision.datasets.utils import download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset



class StanfordCars(datasets.VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        #self._base_folder = pathlib.Path("/home/leofeng/datasets/cars") / "stanford_cars"
        self._base_folder = os.path.join(root, "stanford_cars")
        os.makedirs(self._base_folder, exist_ok=True)
        devkit = os.path.join(self._base_folder, "devkit")

        if self._split == "train":
            self._annotations_mat_path = os.path.join(devkit, "cars_train_annos.mat")
            self._images_base_path = os.path.join(self._base_folder, "cars_train")
        else:
            self._annotations_mat_path = os.path.join(self._base_folder, "cars_test_annos_withlabels.mat")
            self._images_base_path = os.path.join(self._base_folder, "cars_test")

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._samples = [
            (
                os.path.join(self._images_base_path, annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(os.path.join(devkit, "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)


    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target


    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(
            url="https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            download_root=str(self._base_folder),
            md5="c3b158d763b6e2245038c8ad08e45376",
        )
        if self._split == "train":
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
                download_root=str(self._base_folder),
                md5="065e5b463ae28d29e77c1b4b166cfe61",
            )
        else:
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
                download_root=str(self._base_folder),
                md5="4ce7ebf6a94d07f1952d94dd34c4d501",
            )
            download_url(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
                root=str(self._base_folder),
                md5="b0a2b23655a3edd16d84508592a98d10",
            )

    def _check_exists(self) -> bool:
        if not os.path.exists(os.path.join(self._base_folder, "devkit")):
            return False

        return os.path.exists(self._annotations_mat_path) and os.path.isdir(self._images_base_path)

class Cars(StanfordCars):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Cars, self).__init__(
            root, split='train' if train else 'test', transform=transform,
            target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target






#import os
#from PIL import Image
#import numpy as np
#from torch.utils.data import Dataset
#
#class ImageNet100(Dataset):
#    def __init__(self, data_dir, split, transform=None):
#        assert split in ['train', 'val', 'test']
#        self.data_dir = data_dir
#        self.img_path = []
#        self.labels = []
#        self.transform = transform
#
#        if split in ['train', 'val']:
#            imagenet_split = 'train'
#        else:
#            imagenet_split = 'val'
#
#        # The class subset is taken from: https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
#        with open(os.path.join(self.data_dir, 'splits', 'imagenet100.txt')) as f:
#            class_names = list(map(lambda x : x.strip(), f.readlines()))
#
#        num_classes = 100 # Subset of ImageNet
#        for i, class_name in enumerate(class_names):
#            folder_path = os.path.join(
#                self.data_dir, 'imagenet', 'raw', imagenet_split, class_name)
#
#            file_names = os.listdir(folder_path)
#
#            if split is not 'test':
#                num_train = int(len(file_names) * 0.8) # 80% Training data
#
#            for j, fid in enumerate(file_names):
#                if split == 'train' and j >= num_train: # ensures only the first 80% of data is used for training
#                    break
#                elif split == 'val' and j < num_train: # skips the first 80% of data used for training
#                    continue
#                self.img_path.append(os.path.join(folder_path, fid))
#                self.labels.append(i)
#
#        print(f"Dataset Size: {len(self.labels)}")
#        self.targets = self.labels # Sampler needs to use targets
#
#    def __len__(self):
#        return len(self.labels)
#
#    def __getitem__(self, index):
#        path = self.img_path[index]
#        label = self.labels[index]
#
#        with open(path, 'rb') as f:
#            sample = Image.open(f).convert('RGB')
#
#        if self.transform is not None:
#            sample = self.transform(sample)
#
#        return sample, label
