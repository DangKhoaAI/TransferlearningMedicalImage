import numpy as np
import os
import csv
from PIL import Image

def export_image_metadata(folder_path, csv_file):
    try:
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['image_name', 'class', 'width', 'height'])

            for class_name in os.listdir(folder_path):
                class_folder = os.path.join(folder_path, class_name)
                
                if os.path.isdir(class_folder):
                    for image_name in os.listdir(class_folder):
                        image_path = os.path.join(class_folder, image_name)
                        
                        if os.path.isfile(image_path):
                            try:
                                with Image.open(image_path) as img:
                                    width, height = img.size
                                
                                writer.writerow([image_name, class_name, width, height])
                            except Exception as e:
                                print(f"Lỗi khi xử lý ảnh {image_name}: {e}")
        print("Xuất CSV thành công!")
    except Exception as e:
        print(f"Lỗi khi ghi file CSV: {e}")




def export_image_metadata_2(folder_path, output_csv):
    """
    Duyệt qua dataset (Training, Testing) và xuất thông tin ảnh vào file CSV.
    
    Args:
        folder_path (str): Đường dẫn đến thư mục chứa dataset.
        output_csv (str): Tên file CSV đầu ra.
    """
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'class', 'dataset', 'width', 'height'])

        for dataset_type in ['Training', 'Testing']:
            dataset_path = os.path.join(folder_path, dataset_type)
            
            if os.path.isdir(dataset_path):
                for class_name in os.listdir(dataset_path):
                    class_folder = os.path.join(dataset_path, class_name)
                    
                    if os.path.isdir(class_folder):
                        for image_name in os.listdir(class_folder):
                            image_path = os.path.join(class_folder, image_name)
                            if os.path.isfile(image_path):
                                try:
                                    with Image.open(image_path) as img:
                                        width, height = img.size
                                    writer.writerow([image_name, class_name, dataset_type, width, height])
                                except Exception as e:
                                    print(f"Lỗi khi xử lý ảnh {image_name}: {e}")
    print("Xuất CSV thành công!")


# Gọi hàm
export_image_metadata('/kaggle/input/sarscov2-ctscan-dataset', 'covid.csv')
export_image_metadata_2('/kaggle/input/brain-tumor-mri-dataset', 'brain.csv')