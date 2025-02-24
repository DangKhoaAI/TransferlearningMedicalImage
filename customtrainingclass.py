import torch
import torch.nn as nn
import timm
import copy
import time
from torch.utils.data import DataLoader
def train_model(model, train_loader, val_loader, device, num_epochs=10, 
                threshold=0.9, lr_patience=2, stop_patience=4, factor=0.5):
    """
    Huấn luyện model với các kỹ thuật:
      - LR Scheduling: Điều chỉnh learning rate khi không cải thiện theo chỉ số được theo dõi
      - Early Stopping: Dừng training nếu đã điều chỉnh LR quá số lần cho phép
      - Checkpoint: Lưu lại trạng thái model tốt nhất dựa trên validation loss
      
    Tham số:
      - threshold: ngưỡng để lựa chọn metric theo training accuracy hoặc validation loss
      - lr_patience: số epoch không cải thiện trước khi giảm LR
      - stop_patience: số lần điều chỉnh LR liên tiếp không cải thiện trước khi dừng training
      - factor: hệ số nhân để giảm learning rate (new_lr = old_lr * factor)
    """
    # Khai báo loss và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)
    
    # Các list lưu lại kết quả
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    
    # Checkpoint: lưu validation loss tốt nhất (và model state)
    best_val_loss_checkpoint = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    # Các biến dùng cho LR scheduling:
    best_train_acc = 0.0          # theo dõi training accuracy tốt nhất khi train acc < threshold
    best_val_loss_lr = float('inf')  # theo dõi validation loss tốt nhất khi train acc >= threshold
    lr_adjust_counter = 0         # đếm số epoch không cải thiện
    lr_adjust_count = 0           # số lần điều chỉnh LR

    # In header tương tự như log của TensorFlow callback
    header = f'{"Epoch":^6s} {"Loss":^8s} {"Acc":^8s} {"V_loss":^10s} {"V_acc":^8s} {"LR":^10s} {"Next LR":^10s} {"Monitor":^10s} {"%Improv":^10s} {"Duration":^10s}'
    print(header)
    
    total_start = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0
        
        # Huấn luyện theo từng batch
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (outputs.argmax(dim=1) == labels).sum().item()
            total_train += inputs.size(0)
        
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = running_corrects / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Đánh giá trên tập validation
        model.eval()
        running_loss_val = 0.0
        running_corrects_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss_val += loss.item() * inputs.size(0)
                running_corrects_val += (outputs.argmax(dim=1) == labels).sum().item()
                total_val += inputs.size(0)
                
        epoch_val_loss = running_loss_val / total_val
        epoch_val_acc = running_corrects_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Lấy learning rate hiện tại trước khi điều chỉnh
        current_lr = optimizer.param_groups[0]['lr']
        next_lr = current_lr  # mặc định sẽ không thay đổi
        monitor = ''
        improvement = 0.0  # % cải thiện so với giá trị tốt nhất trước đó
        
        # Nếu training accuracy dưới threshold thì theo dõi training accuracy,
        # ngược lại thì theo dõi validation loss.
        if epoch_train_acc < threshold:
            monitor = 'accuracy'
            if best_train_acc > 0:
                improvement = (epoch_train_acc - best_train_acc) * 100 / best_train_acc
            else:
                improvement = 0.0
            if epoch_train_acc > best_train_acc:
                best_train_acc = epoch_train_acc
                lr_adjust_counter = 0
                # Cập nhật checkpoint nếu validation loss cũng cải thiện
                if epoch_val_loss < best_val_loss_checkpoint:
                    best_val_loss_checkpoint = epoch_val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1
            else:
                lr_adjust_counter += 1
                if lr_adjust_counter >= lr_patience:
                    next_lr = current_lr * factor
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = next_lr
                    print(f"Reducing LR from {current_lr:.6f} to {next_lr:.6f} due to lack of train accuracy improvement.")
                    lr_adjust_counter = 0
                    lr_adjust_count += 1
        else:
            monitor = 'val_loss'
            if best_val_loss_lr < float('inf'):
                improvement = (best_val_loss_lr - epoch_val_loss) * 100 / best_val_loss_lr
            else:
                improvement = 0.0
            if epoch_val_loss < best_val_loss_lr:
                best_val_loss_lr = epoch_val_loss
                lr_adjust_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
            else:
                lr_adjust_counter += 1
                if lr_adjust_counter >= lr_patience:
                    next_lr = current_lr * factor
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = next_lr
                    print(f"Reducing LR from {current_lr:.6f} to {next_lr:.6f} due to lack of validation loss improvement.")
                    lr_adjust_counter = 0
                    lr_adjust_count += 1
        
        # Early stopping nếu số lần điều chỉnh LR vượt quá giới hạn
        if lr_adjust_count >= stop_patience:
            print(f"Early stopping: LR adjusted {lr_adjust_count} times without improvement.")
            break
        
        epoch_duration = time.time() - epoch_start
        # In log chi tiết cho epoch này
        print(f"{epoch+1:^6d} {epoch_train_loss:^8.3f} {epoch_train_acc*100:^8.3f} {epoch_val_loss:^10.5f} {epoch_val_acc*100:^8.3f} {current_lr:^10.5f} {next_lr:^10.5f} {monitor:^10s} {improvement:^10.2f} {epoch_duration:^10.2f}")
    
    total_duration = time.time() - total_start
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = total_duration % 60
    print(f"Total training time: {hours}h {minutes}m {seconds:.2f}s")
    
    # Sau khi huấn luyện, load lại trạng thái model tốt nhất
    model.load_state_dict(best_model_state)
    print(f"Best model from epoch {best_epoch} restored.")

    return train_losses, train_accs, val_losses, val_accs