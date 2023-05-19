from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


class InteriorDataset(Dataset):
    def __init__(self, img_dir, annotations_file=None, transform=None, target_transform=None) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.img_paths = []
        self.annotations_file = annotations_file
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)
        return image
