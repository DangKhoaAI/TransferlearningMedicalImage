import pandas as pd
import torch

def save_training_history(train_losses, train_accs, val_losses, val_accs, dataset_name, model_name):
    file_name = f"{dataset_name}-{model_name}-history.csv"
    df = pd.DataFrame({
        "Epoch": list(range(1, len(train_losses) + 1)),
        "Train Loss": train_losses,
        "Train Accuracy": [acc * 100 for acc in train_accs],  # Chuyển acc thành %
        "Validation Loss": val_losses,
        "Validation Accuracy": [acc * 100 for acc in val_accs]
    })
    df.to_csv(file_name, index=False)
    print(f"✅ Training results saved to {file_name}")

def save_evaluation_results(train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, dataset_name, model_name):
    file_name = f"{dataset_name}-{model_name}-evaluation.txt"
    with open(file_name, "w") as f:
        f.write(f"Train Loss: {train_loss:.4f}\n")
        f.write(f"Train Accuracy: {train_acc:.2f}%\n")
        f.write("-" * 20 + "\n")
        f.write(f"Validation Loss: {val_loss:.4f}\n")
        f.write(f"Validation Accuracy: {val_acc:.2f}%\n")
        f.write("-" * 20 + "\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
    print(f"✅ Đã lưu evaluation vào {file_name}")
