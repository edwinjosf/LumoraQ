import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

def get_dataloaders(batch_size=32, augment=True):
    transform = get_transforms(augment=augment)
    dataset = datasets.EuroSAT(root=DATA_DIR, download=True, transform=transform)

    n = len(dataset)
    train_n = int(0.7 * n)
    val_n   = int(0.15 * n)
    test_n  = n - train_n - val_n

    train_set, val_set, test_set = random_split(
        dataset, [train_n, val_n, test_n]
    )

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2),
        DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2),
    )
