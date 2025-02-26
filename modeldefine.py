import torch
import torch.nn as nn
import timm

class MyModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(MyModel, self).__init__()
        self.base_model = base_model

        # Lấy số feature cuối cùng
        self.embed_dim = base_model.num_features

        # Tạo classifier
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        features = self.base_model.forward_features(x)  # Hoạt động cho cả ViT & ResNet
        
        # Nếu là ViT, feature shape (B, num_tokens, embed_dim) → Lấy mean trên token
        if features.dim() == 3: 
            pooled = features.mean(dim=1)
        else:  # CNN có shape (B, C, H, W) → Dùng global average pooling
            pooled = features.mean(dim=[2, 3])     
        logits = self.classifier(pooled)
        return logits
def get_model(num_classes, basemodel, device, freeze=False):
    base_model = timm.create_model(basemodel, pretrained=True, num_classes=0) 
    if freeze==True:
        for param in base_model.parameters():
            param.requires_grad = False
    
    model = MyModel(base_model, num_classes).to(device)
    return model