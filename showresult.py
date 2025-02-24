from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(train_losses, train_accs, val_losses, val_accs,filename='trainingprocess.png'):
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, marker='o', label='Train Loss')
    plt.plot(val_losses, marker='o', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.plot([acc*100 for acc in train_accs], marker='o', label='Train Accuracy')
    plt.plot([acc*100 for acc in val_accs], marker='o', label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"✅ Training process image saved to {filename}")
    plt.show()
def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)
    test_loss = running_loss / total
    test_acc = (correct / total)*100
    #print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc*100:.2f}%")
    return test_loss , test_acc

def evaluate_model(model, data_loader, class_names, img_name='confusion-matrix.png',file_name='classificationreport.txt'):
    """
    Hàm này thực hiện dự đoán, tạo confusion matrix và classification report.
    
    Args:
        model: Mô hình đã train.
        data_loader: DataLoader chứa dữ liệu test.
        class_names: Danh sách tên các lớp.
        filename: Tên file để lưu confusion matrix.
    """
    # Bước 1: Predict
    model.eval()  # Chuyển sang chế độ đánh giá
    y_pred, y_true = [], []

    with torch.no_grad():  
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Chuyển về GPU nếu có
            
            outputs = model(inputs)  # Dự đoán logits
            preds = torch.argmax(outputs, dim=1)  # Lấy class có xác suất cao nhất

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    # Bước 2: Tạo confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]}", horizontalalignment='center', 
                 color='white' if cm[i, j] > thresh else 'black')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(img_name)
    print(f"✅ Confusion matrix image saved to {img_name}")
    plt.show() 
    # Bước 3: In classification report
    report = sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    with open(file_name, "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm) + "\n")
        f.write("\nClassification Report:\n")
        f.write(report) 
    print(f"✅ Classification report saved to {file_name}")
    return y_pred, cm, report
