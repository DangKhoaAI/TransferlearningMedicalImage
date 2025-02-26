import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sử dụng seaborn style để biểu đồ chuyên nghiệp hơn
sns.set_style("whitegrid")

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

def process_and_plot(paths, model_names, title):
    """Đọc dữ liệu từ file và vẽ biểu đồ so sánh Accuracy."""
    train_acc, val_acc, test_acc = [], [], []
    
    for file in paths:
        _, ta, _, va, _, tsa = extract_metrics(file)
        train_acc.append(ta)
        val_acc.append(va)
        test_acc.append(tsa)
    
    x = np.arange(len(model_names))  # Vị trí trên trục x
    width = 0.2  # Độ rộng cột

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Chọn màu sắc chuyên nghiệp
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Xanh, cam, xanh lá

    # Vẽ bar chart
    bars1 = ax.bar(x - width, train_acc, width, label="Train Accuracy", color=colors[0], alpha=0.85)
    bars2 = ax.bar(x, val_acc, width, label="Validation Accuracy", color=colors[1], alpha=0.85)
    bars3 = ax.bar(x + width, test_acc, width, label="Test Accuracy", color=colors[2], alpha=0.85)

    # Ghi giá trị lên cột
    def annotate_bars(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),  # Dịch lên trên một chút
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    annotate_bars(bars1)
    annotate_bars(bars2)
    annotate_bars(bars3)

    # Cấu hình trục x, y
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # Hiển thị grid nhẹ
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()  # Đảm bảo bố cục gọn gàng
    plt.show()

if __name__=="__main__":
    # Danh sách file và mô hình tương ứng
    covid_paths= [
        'Covid-ConvNeXt-evaluation.txt', 
        'Covid-DenseNet-evaluation.txt',
        'Covid-ResNet-evaluation.txt', 
        'Covid-ViT-evaluation.txt', 
        'Covid-ViT+-evaluation.txt'
                ]
    lungcancer_paths= [
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

    # Gọi hàm để vẽ biểu đồ
    covid_title="Comparison of Accuracy Among Models on the SARS-COV-2 Dataset"
    process_and_plot(covid_paths, model_names, covid_title)

    lungcancer_title="Comparison of Accuracy Among Models on the Lungcancer Dataset"
    process_and_plot(lungcancer_paths, model_names, lungcancer_title)

    braintumor_title="Comparison of Accuracy Among Models on the Brain Tumor MRI Dataset"
    process_and_plot(braintumor_paths, model_names, braintumor_title)
