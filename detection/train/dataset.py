from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_datasets(trainset_dir: str, testset_dir: str):
    # transform
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # datasets
    trainset = ImageFolder(root=trainset_dir, transform=train_transform)
    testset = ImageFolder(root=testset_dir, transform=test_transform)
    
    return trainset, testset
