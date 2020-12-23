import torchvision as Tv
import torch

def load_data():
    transforms = Tv.transforms.Compose([
        Tv.transforms.Resize((256, 256)),
        # Tv.transforms.RandomCrop((256, 256)),
        Tv.transforms.CenterCrop((256, 256)),
        Tv.transforms.ToTensor(),
        Tv.transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
    ])
    data1 = Tv.datasets.ImageFolder(input("Where is the data #1?:"), 
                            transform=transforms)

    data2 = Tv.datasets.ImageFolder(input("Where is the data #2?:"), 
                            transform=transforms)
    merged = torch.utils.data.ConcatDataset([data1, data2])

    dataloader = torch.utils.data.DataLoader(data1, shuffle=True, num_workers=4, batch_size=1, drop_last=True)
    dataloader2 = torch.utils.data.DataLoader(data2, shuffle=True, num_workers=4, batch_size=1, drop_last=True)
    return dataloader, dataloader2