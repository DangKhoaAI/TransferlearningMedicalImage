from training import train_model
from dataloader import get_transforms_timm, load_dataset, get_dataloaders  
from modeldefine import MyModel ,get_model
from showresult import plot_metrics, test_model, evaluate_model
from saving import save_training_history, save_evaluation_results
from torchinfo import summary
import torch
### define model ,data ,hyperparameter
dataset_path= '/kaggle/input/lung-cancer-histopathological-images' #dataset path in kaggle
model_code=   'convnext_tiny.fb_in22k'#model code in timm
dataset_name='Lungcancer' #name of dataset to saving
model_name='ConvNeXt' #name of model to saving
num_epochs=10 #number of epochs to run (data1=30 , data2=10 ,data3=)
lr_patience=2 #patience before edit lr (data1=3 , data2=2)
stop_patience=2 # times edit lr befor stop training (data1=3,data2=2)a
is_freeze=False #model if freeze (not train parameter) or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device to train CPU /GPU
split_ratio=[0.8, 0.1, 0.1] # ratio of train-test-val 
batch_size=32 # batch size of data (data1=32)

if __name__=="__main__":
    #data
    train_transform = get_transforms_timm(model_code, is_training=True)
    val_transform = get_transforms_timm(model_code, is_training=False)
    train_set, val_set, test_set,num_classes = load_dataset(dataset_path, train_transform, val_transform,split_ratio=split_ratio)
    train_loader, val_loader, test_loader = get_dataloaders(train_set, val_set, test_set,batch_size=batch_size)
    #model 
    model = get_model(num_classes, model_code, device, freeze=is_freeze)
    summary(model, input_size=(1, 3, 224, 224))
    #training
    train_losses, train_accs, val_losses, val_accs = train_model(model, train_loader, val_loader, device,num_epochs=num_epochs,threshold=0.9,lr_patience=lr_patience,stop_patience=stop_patience)
    
    # show results , saving result
    #a.history
    save_training_history(train_losses=train_losses, train_accs=train_accs, val_losses=val_losses, val_accs=val_accs,
                          dataset_name=dataset_name,model_name=model_name)

    #b.history plot
    plot_name = f"{dataset_name}-{model_name}-trainingprocess.png"
    plot_metrics(train_losses, train_accs, val_losses, val_accs,plot_name)

    #c.evaluation
    train_loss, train_acc = test_model(model, train_loader)
    val_loss, val_acc = test_model(model, val_loader)
    test_loss, test_acc = test_model(model, test_loader)
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print('-' * 20)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print('-' * 20)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    save_evaluation_results(train_losses=train_losses, train_accs=train_accs, val_losses=val_losses, val_accs=val_accs,
                          dataset_name=dataset_name,model_name=model_name)
    

    #d.comfusion matrix,classification report
    cm_img_name = f"{dataset_name}-{model_name}-confusion-matrix.png"
    creport_name = f"{dataset_name}-{model_name}-classification_report.txt"
    class_names = test_loader.dataset.dataset.classes
    y_pred, cm, report = evaluate_model(model, test_loader, class_names,cm_img_name,creport_name)
