import re
import pandas as pd
import matplotlib.pyplot as plt
# Hàm đọc và trích xuất macro avg từ file classification report
def extract_macro_avg(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    # Regex tìm dòng chứa macro avg
    match = re.search(r"macro avg\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)", content)
    if match:
        return map(float, match.groups())  # Trả về (precision, recall, f1-score)
    return None, None, None  # Trả về None nếu không tìm thấy

# Hàm tạo bảng DataFrame từ danh sách file
def create_comparison_table(file_paths, model_names):
    precision, recall, f1_score = [], [], []

    # Duyệt từng file, trích xuất dữ liệu
    for file in file_paths:
        p, r, f1 = extract_macro_avg(file)
        precision.append(p)
        recall.append(r)
        f1_score.append(f1)

    # Tạo DataFrame
    df = pd.DataFrame({
        "Model": model_names,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score
    })

    return df
def save_table_as_image(df, title, filename):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)  # Kích thước lớn hơn, độ phân giải cao
    ax.set_frame_on(False)  # Tắt viền xung quanh
    ax.axis('off')  # Tắt trục

    # Định dạng màu nền xen kẽ
    cell_colors = [["#f5f5f5" if i % 2 == 0 else "#ffffff" for _ in df.columns] for i in range(len(df))]

    # Tạo bảng với style đẹp
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors
    )

    # Định dạng tiêu đề cột (header)
    for i, cell in enumerate(table._cells[(0, j)] for j in range(len(df.columns))):
        cell.set_fontsize(12)
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4a4a4a')  # Header màu tối

    # Định dạng từng ô trong bảng
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 1.3)  # Phóng to bảng

    # Đặt tiêu đề bảng
    ax.set_title(title, fontsize=14, fontweight="bold", fontname="serif", pad=10)

    # Lưu hình ảnh
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
### SAVING TO TXT
# Hàm lưu bảng vào file TXT
def save_table_to_txt(df, filename):
    with open(filename, "w") as f:
        f.write(df.to_string(index=False))  # Ghi DataFrame vào file TXT
if __name__=="__main__":
    # Danh sách file TXT và mô hình tương ứng
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

    # In bảng kết quả
    print("=== COVID Dataset ===")
    print(covid_df)
    print("\n=== Lung Cancer Dataset ===")
    print(lungcancer_df)
    print("\n=== Brain Tumor Dataset ===")
    print(braintumor_df)




    # Lưu bảng vào file
    #save_table_to_txt(covid_df, "covid_classification_report.txt")
    #save_table_to_txt(lungcancer_df, "lungcancer_classification_report.txt")
    #save_table_to_txt(braintumor_df, "braintumor_classification_report.txt")

    # Xuất bảng dưới dạng hình ảnh
    #save_table_as_image(covid_df, "SARS-COV-2 Classification Report", "covid_report.png")
    #save_table_as_image(lungcancer_df, "Lung Cancer Classification Report", "lungcancer_report.png")
    #save_table_as_image(braintumor_df, "Brain Tumor Classification Report", "braintumor_report.png")
