import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sử dụng seaborn style cho biểu đồ chuyên nghiệp
sns.set_style("whitegrid")

def extract_metrics(file_path):
    """
    Đọc và trích xuất metrics từ file TXT (đánh giá model).
    Trích xuất: Train Loss, Train Accuracy, Validation Loss, Validation Accuracy, Test Loss, Test Accuracy.
    """
    with open(file_path, "r") as file:
        content = file.read()
    
    train_loss = float(re.search(r"Train Loss: ([\d\.]+)", content).group(1))
    train_acc = float(re.search(r"Train Accuracy: ([\d\.]+)%", content).group(1))
    val_loss = float(re.search(r"Validation Loss: ([\d\.]+)", content).group(1))
    val_acc = float(re.search(r"Validation Accuracy: ([\d\.]+)%", content).group(1))
    test_loss = float(re.search(r"Test Loss: ([\d\.]+)", content).group(1))
    test_acc = float(re.search(r"Test Accuracy: ([\d\.]+)%", content).group(1))
    
    return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc

def time_str_to_seconds(time_str):
    """
    Chuyển đổi chuỗi thời gian dạng '0h 4m 26.70s' thành số giây.
    """
    hours = re.search(r"(\d+)h", time_str)
    minutes = re.search(r"(\d+)m", time_str)
    seconds = re.search(r"([\d\.]+)s", time_str)
    h = int(hours.group(1)) if hours else 0
    m = int(minutes.group(1)) if minutes else 0
    s = float(seconds.group(1)) if seconds else 0
    total_seconds = h * 3600 + m * 60 + s
    return total_seconds

def extract_training_times(file_path, dataset_keyword):
    """
    Đọc file trainingtime.txt và trích xuất training time của các model cho dataset có tên chứa dataset_keyword.
    Ví dụ: nếu dataset_keyword là "Brain", sẽ trích xuất các dòng bắt đầu bằng "Brain - ...".
    Trả về một dictionary: { model_name: training_time (seconds) }
    """
    training_times = {}
    with open(file_path, "r") as file:
        content = file.read()
    
    # Tách các section dựa trên dấu '####'
    sections = re.split(r"####+", content)
    dataset_section = None
    for section in sections:
        if dataset_keyword in section:
            dataset_section = section
            break
    
    if dataset_section is None:
        print(f"Không tìm thấy dataset '{dataset_keyword}' trong file trainingtime.txt.")
        return training_times
    
    # Mẫu dòng: "Brain - ResNet : Early Stopiing, Total training time: 0h 4m 34.14s .Best model from epoch 5 restored."
    pattern = re.compile(rf"{dataset_keyword}\s*-\s*([^:]+):.*Total training time:\s*([\d]+h\s*[\d]+m\s*[\d\.]+s)", re.IGNORECASE)
    matches = pattern.findall(dataset_section)
    
    for match in matches:
        model_name = match[0].strip()
        time_str = match[1].strip()
        seconds = time_str_to_seconds(time_str)
        training_times[model_name] = seconds
    return training_times

def scatter_plot_time_accuracy(evaluation_paths, training_time_file, dataset_keyword, model_names, title):
    """
    Vẽ scatter plot cho mối quan hệ giữa Training Time (x-axis) và Test Accuracy (y-axis) của các model.
    
    evaluation_paths : danh sách file đánh giá (mỗi file chứa metrics của 1 model)
    training_time_file : file chứa thông tin training time
    dataset_keyword : tên dataset cần trích xuất trong file training time (ví dụ: "Brain")
    model_names : danh sách tên model theo thứ tự muốn hiển thị (ví dụ: ["ConvNeXt", "DenseNet", "ResNet", "ViT", "ViT*"])
    title : tiêu đề cho biểu đồ
    """
    # Lấy Test Accuracy từ các file đánh giá
    test_accuracies = []
    for file in evaluation_paths:
        _, _, _, _, _, test_acc = extract_metrics(file)
        test_accuracies.append(test_acc)
    
    # Lấy training time từ file trainingtime.txt
    training_times_dict = extract_training_times(training_time_file, dataset_keyword)
    
    # Do tên model trong file training time có thể khác với tên dùng ở evaluation,
    # ta tạo mapping: model tên hiển thị => tên trong trainingtime.txt
    mapping = {
        "ConvNeXt": "ConvNeXt",
        "DenseNet": "DenseNet",
        "ResNet": "ResNet",
        "ViT": "ViT",
        "ViT*": "ViT+"
    }
    
    # Tạo danh sách training time theo thứ tự model_names
    training_times = []
    for m in model_names:
        key = mapping.get(m, m)
        if key in training_times_dict:
            training_times.append(training_times_dict[key])
        else:
            training_times.append(None)
            print(f"Không tìm thấy training time cho model {m} (key: {key}).")
    
    # Lọc ra các model có đủ dữ liệu (không None)
    filtered_models = []
    filtered_train_times = []
    filtered_accuracies = []
    for i, t in enumerate(training_times):
        if t is not None:
            filtered_models.append(model_names[i])
            filtered_train_times.append(t)
            filtered_accuracies.append(test_accuracies[i])
    
    # Vẽ scatter plot với màu sắc khác nhau cho từng model
    colors = ['#1f77b4',  '#2ca02c',  '#9467bd',  '#ff7f0e',  '#d62728']

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(filtered_models):
        plt.scatter(filtered_train_times[i], filtered_accuracies[i], 
                    color=colors[i % len(colors)], s=100, label=model)

    # Ghi tên model bên cạnh từng điểm dữ liệu
    for i, model in enumerate(filtered_models):
        plt.annotate(model, (filtered_train_times[i], filtered_accuracies[i]),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=10, fontweight='bold')
    
    plt.xlabel("Training Time (seconds)", fontsize=12, fontweight='bold')
    plt.ylabel("Test Accuracy (%)", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ---------------------------
# MAIN: CÁC THAM SỐ VÀ GỌI HÀM
# ---------------------------
if __name__=="__main__":
    # Danh sách file đánh giá cho dataset Brain
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

    # Danh sách tên model theo thứ tự muốn hiển thị
    model_names = ["ConvNeXt", "DenseNet", "ResNet", "ViT", "ViT*"]

    # File chứa training time
    training_time_file = "trainingtime.txt"
    
    

    # Tiêu đề cho scatter plot
    # Chọn dataset cần hiển thị (trong file trainingtime.txt có các section: COVID, Lung, Brain)
    covid_keyword = "COVID"
    covid_title = "Scatter Plot: Training Time vs Test Accuracy SARS-COV-2 Dataset"
    scatter_plot_time_accuracy(braintumor_paths, training_time_file, covid_keyword, model_names, covid_title)

    lungcancer_keyword="Lung"
    lungcancer_title = "Scatter Plot: Training Time vs Test Accuracy Lung Cancer Dataset"
    scatter_plot_time_accuracy(braintumor_paths, training_time_file, lungcancer_keyword, model_names, lungcancer_title)

    braintumor_keyword="Brain"
    braintumor_title = "Scatter Plot: Training Time vs Test Accuracy Brain Tumor Dataset"
    scatter_plot_time_accuracy(braintumor_paths, training_time_file, braintumor_keyword, model_names, braintumor_title)
