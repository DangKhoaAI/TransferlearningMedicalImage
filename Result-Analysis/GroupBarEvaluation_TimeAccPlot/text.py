import re

def extract_metrics(file_path):
    """Đọc và trích xuất metrics từ file TXT."""
    with open(file_path, "r") as file:
        content = file.read()
    
    train_loss = float(re.search(r"Train Loss: ([\d\.]+)", content).group(1))
    train_acc = float(re.search(r"Train Accuracy: ([\d\.]+)%", content).group(1))
    val_loss = float(re.search(r"Validation Loss: ([\d\.]+)", content).group(1))
    val_acc = float(re.search(r"Validation Accuracy: ([\d\.]+)%", content).group(1))
    test_loss = float(re.search(r"Test Loss: ([\d\.]+)", content).group(1))
    test_acc = float(re.search(r"Test Accuracy: ([\d\.]+)%", content).group(1))
    
    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc

def process_and_write(paths, model_names, title, output_file):
    """Đọc dữ liệu từ file và xuất ra file text."""
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\n{title}\n")
        f.write("=" * len(title) + "\n")
        
        for file, model in zip(paths, model_names):
            train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = extract_metrics(file)
            f.write(f"\nModel: {model}\n")
            f.write(f"  Train Loss        : {train_loss}\n")
            f.write(f"  Train Accuracy    : {train_acc}%\n")
            f.write(f"  Validation Loss   : {val_loss}\n")
            f.write(f"  Validation Accuracy: {val_acc}%\n")
            f.write(f"  Test Loss         : {test_loss}\n")
            f.write(f"  Test Accuracy     : {test_acc}%\n")
            f.write("-" * 40 + "\n")

if __name__=="__main__":
    # Danh sách file và tên model tương ứng
    covid_paths = [
        'Covid-ConvNeXt-evaluation.txt', 
        'Covid-DenseNet-evaluation.txt',
        'Covid-ResNet-evaluation.txt', 
        'Covid-ViT-evaluation.txt', 
        'Covid-ViT+-evaluation.txt'
    ]
    lungcancer_paths = [
        'Lungcancer-ConvNeXt-evaluation.txt',
        'Lungcancer-DenseNet-evaluation.txt', 
        'Lungcancer-ResNet-evaluation.txt', 
        'Lungcancer-ViT-evaluation.txt', 
        'Lungcancer-ViT+-evaluation.txt'
    ]
    braintumor_paths = [
        'Braintumor-raw-ConvNeXt-evaluation.txt',
        'Braintumor-raw-DenseNet-evaluation.txt',
        'Braintumor-raw-ResNet-evaluation.txt',
        'Braintumor-raw-ViT-evaluation.txt',
        'Braintumor-raw-ViT+-evaluation.txt'
    ]
    model_names = ["ConvNeXt", "DenseNet", "ResNet", "ViT", "ViT*"]

    # Đường dẫn file xuất kết quả
    output_file = "evaluation_results.txt"
    
    # Xoá nội dung file nếu đã tồn tại để ghi mới từ đầu
    open(output_file, "w", encoding="utf-8").close()
    
    # Xuất kết quả cho từng tập dữ liệu
    covid_title = "Comparison of Accuracy Among Models on the SARS-COV-2 Dataset"
    process_and_write(covid_paths, model_names, covid_title, output_file)

    lungcancer_title = "Comparison of Accuracy Among Models on the Lungcancer Dataset"
    process_and_write(lungcancer_paths, model_names, lungcancer_title, output_file)

    braintumor_title = "Comparison of Accuracy Among Models on the Brain Tumor MRI Dataset"
    process_and_write(braintumor_paths, model_names, braintumor_title, output_file)
