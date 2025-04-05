import re
import pandas as pd

# Hàm đọc và trích xuất macro avg từ file classification report
def extract_macro_avg(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    # Regex tìm dòng chứa macro avg (precision, recall, f1-score)
    match = re.search(r"macro avg\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)", content)
    if match:
        return map(float, match.groups())
    return None, None, None

# Hàm tạo bảng DataFrame từ danh sách file
def create_comparison_table(file_paths, model_names):
    precision, recall, f1_score = [], [], []

    # Duyệt từng file, trích xuất dữ liệu
    for file in file_paths:
        p, r, f1 = extract_macro_avg(file)
        precision.append(p)
        recall.append(r)
        f1_score.append(f1)

    # Tạo DataFrame chứa kết quả
    df = pd.DataFrame({
        "Model": model_names,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score
    })
    return df

# Hàm lưu các bảng kết quả vào 1 file TXT tổng hợp
def save_all_tables_to_txt(covid_df, lungcancer_df, braintumor_df, filename):
    with open(filename, "w") as f:
        f.write("=== COVID Dataset ===\n")
        f.write(covid_df.to_string(index=False))
        f.write("\n\n=== Lung Cancer Dataset ===\n")
        f.write(lungcancer_df.to_string(index=False))
        f.write("\n\n=== Brain Tumor Dataset ===\n")
        f.write(braintumor_df.to_string(index=False))

if __name__ == "__main__":
    # Danh sách file TXT cho từng dataset và tên mô hình tương ứng
    covid_paths = [
        'Covid-ConvNeXt-classification_report.txt', 
        'Covid-DenseNet-classification_report.txt',
        'Covid-ResNet-classification_report.txt', 
        'Covid-ViT-classification_report.txt', 
        'Covid-ViT+-classification_report.txt'
    ]

    lungcancer_paths = [
        'Lungcancer-ConvNeXt-classification_report.txt',
        'Lungcancer-DenseNet-classification_report.txt', 
        'Lungcancer-ResNet-classification_report.txt', 
        'Lungcancer-ViT-classification_report.txt', 
        'Lungcancer-ViT+-classification_report.txt'
    ]

    braintumor_paths = [
        'Braintumor-raw-ConvNeXt-classification_report.txt',
        'Braintumor-raw-DenseNet-classification_report.txt',
        'Braintumor-raw-ResNet-classification_report.txt',
        'Braintumor-raw-ViT-classification_report.txt',
        'Braintumor-raw-ViT+-classification_report.txt'
    ]
    
    model_names = ["ConvNeXt", "DenseNet", "ResNet", "ViT", "ViT*"]
    
    # Tạo bảng so sánh cho từng dataset
    covid_df = create_comparison_table(covid_paths, model_names)
    lungcancer_df = create_comparison_table(lungcancer_paths, model_names)
    braintumor_df = create_comparison_table(braintumor_paths, model_names)

    # In bảng kết quả ra console
    print("=== COVID Dataset ===")
    print(covid_df)
    print("\n=== Lung Cancer Dataset ===")
    print(lungcancer_df)
    print("\n=== Brain Tumor Dataset ===")
    print(braintumor_df)

    # Lưu tất cả thông tin trích xuất vào file TXT tổng hợp
    save_all_tables_to_txt(covid_df, lungcancer_df, braintumor_df, "all_classification_reports.txt")
