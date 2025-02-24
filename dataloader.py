import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
# -------------------------
# 1. Chuẩn bị transform
# -------------------------
def get_transforms_timm(model_name, is_training=True):
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.eval()  # Chế độ inference
    data_config = resolve_model_data_config(model)  # Lấy config phù hợp với model
    transform = create_transform(**data_config, is_training=is_training)
    return transform
# -------------------------
# 2. Load dataset và chia tập
# -------------------------
def load_dataset(dataset_path, train_transform, val_transform, split_ratio=[0.8, 0.1, 0.1]):
    # Tạo dataset đầy đủ với transform của train (mặc định)
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)
    dataset_size = len(full_dataset)
    
    
    if len(split_ratio) == 2:
        train_ratio, val_ratio = split_ratio
        test_ratio = 1.0 - (train_ratio + val_ratio)
        ratios = [train_ratio, val_ratio, test_ratio]
    elif len(split_ratio) == 3:
        ratios = split_ratio
    else:
        ratios = [0.8, 0.1, 0.1]
    
    # Tính số lượng mẫu cho từng tập theo tỷ lệ
    sizes = [int(r * dataset_size) for r in ratios]
    # Điều chỉnh kích thước của tập cuối (test) để đảm bảo tổng bằng dataset_size
    sizes[-1] = dataset_size - sum(sizes[:-1])
    
    # Phân chia dataset thành 3 phần: train, validation và test
    train_set, val_set, test_set = random_split(full_dataset, sizes)
    
    # Gán transform mới cho tập validation và test
    val_set.dataset.transform = val_transform
    test_set.dataset.transform = val_transform
    
    return train_set, val_set, test_set, len(full_dataset.classes)
def get_dataloaders(train_set, val_set, test_set, batch_size=32):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader