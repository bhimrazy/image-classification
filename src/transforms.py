from torchvision import transforms as T

train_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        # T.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],  # mean
            [0.229, 0.224, 0.225],  # std
        ),
    ]
)

test_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],  # mean
            [0.229, 0.224, 0.225],  # std
        ),
    ]
)
