import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_multiple_trainings(csv_files, models, data_name,colors):
    '''
    Hàm này vẽ lịch sử training của nhiều mô hình từ các file CSV với format chuyên nghiệp hơn.
    '''
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")

    # Subplot 1: Training & Validation Loss
    plt.subplot(2, 1, 1)
    plt.title('Training and Validation Loss '+data_name, fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    # Subplot 2: Training & Validation Accuracy
    plt.subplot(2, 1, 2)
    plt.title('Training and Validation Accuracy '+data_name, fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        history_dict = {
            'accuracy': df['Train Accuracy'].tolist(),
            'loss': df['Train Loss'].tolist(),
            'val_accuracy': df['Validation Accuracy'].tolist(),
            'val_loss': df['Validation Loss'].tolist()
        }

        index_loss = np.argmin(history_dict['val_loss'])
        val_lowest = history_dict['val_loss'][index_loss]
        index_acc = np.argmax(history_dict['val_accuracy'])
        acc_highest = history_dict['val_accuracy'][index_acc]

        epochs = list(range(1, len(history_dict['accuracy']) + 1))

        # Plot Loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs, history_dict['loss'], linestyle='--', linewidth=1.5, color=colors[i], label=f'Train {models[i]}')
        plt.plot(epochs, history_dict['val_loss'], linestyle='-', linewidth=1.5, color=colors[i], label=f'Val {models[i]}')
        plt.scatter(index_loss + 1, val_lowest, s=80, color=colors[i], edgecolor='black', marker='o', label=f'Best Model {models[i]}')

        # Plot Accuracy
        plt.subplot(2, 1, 2)
        plt.plot(epochs, history_dict['accuracy'], linestyle='--', linewidth=1.5, color=colors[i], label=f'Train {models[i]}')
        plt.plot(epochs, history_dict['val_accuracy'], linestyle='-', linewidth=1.5, color=colors[i], label=f'Val {models[i]}')
        plt.scatter(index_acc + 1, acc_highest, s=80, color=colors[i], edgecolor='black', marker='o', label=f'Best Model {models[i]}')

    # Set legend
    plt.subplot(2, 1, 1)
    plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.2, 1))

    plt.subplot(2, 1, 2)
    plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.2, 1))

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    # Danh sách các file CSV
    model_names = ["ConvNeXt", "DenseNet", "ResNet", "ViT", "ViT*"]
    # Danh sách file TXT và mô hình tương ứng
    covid_paths = [
    'Covid-ConvNeXt-history.csv', 
    'Covid-DenseNet-history.csv',
    'Covid-ResNet-history.csv', 
    'Covid-ViT-history.csv', 
    'Covid-ViT+-history.csv'
    ]
    lungcancer_paths = [
    'Lungcancer-ConvNeXt-history.csv',
    'Lungcancer-DenseNet-history.csv', 
    'Lungcancer-ResNet-history.csv', 
    'Lungcancer-ViT-history.csv', 
    'Lungcancer-ViT+-history.csv'
    ]
    braintumor_paths = [
    'Braintumor-raw-ConvNeXt-history.csv',
    'Braintumor-raw-DenseNet-history.csv',
    'Braintumor-raw-ResNet-history.csv',
    'Braintumor-raw-ViT-history.csv',
    'Braintumor-raw-ViT+-history.csv'
    ]

    colors = ['#1f77b4',  '#2ca02c',  '#9467bd',  '#ff7f0e',  '#d62728']

    # Gọi hàm để plot
    plot_multiple_trainings(covid_paths, model_names,data_name='SARS-COV-2', colors=colors)
    plot_multiple_trainings(lungcancer_paths, model_names,data_name='Lung cancer', colors=colors)
    plot_multiple_trainings(braintumor_paths, model_names,data_name='Brain tumor', colors=colors)