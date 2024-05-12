import os
import torch
import tempfile
import random as r
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import mlflow
from mlflow_utils import create_mlflow_experiment

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import lightning as L


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0) 
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16*4*4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        
        x = self.pool(self.relu(self.conv1(x))) # 1x28x28 -> 6x24x24 -> 6x12x12
        x = self.pool(self.relu(self.conv2(x))) # 6x12x12 -> 16x8x8 -> 16x4x4
        x = x.view(-1, 16*4*4) # 16x4x4 -> 256
        x = self.relu(self.fc1(x)) # 256 -> 64
        x = self.relu(self.fc2(x)) # 64 -> 32
        x = self.fc3(x) # 32 -> 10
        return x
    
    
train_dataset = ImageFolder("data/training", transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
]))

test_dataset = ImageFolder("data/testing", transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
]))



def train_epoch(model:nn.Module, device:str, train_loader:DataLoader, criterion, optimizer):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to be trained.
        device (str): The device to be used for training ("cuda" or "cpu").
        train_loader (DataLoader): The data loader for the training dataset.
        criterion: The loss function.
        optimizer: The optimizer.

    Returns:
        float: The average training loss for the epoch.
        float: The average training accuracy for the epoch.
    """
    model.train()
    train_loss_epoch = 0
    train_acc_epoch = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()
        train_acc_epoch += (y_pred.argmax(1) == y).sum().item() # Convert the tensor to scalar
    return train_loss_epoch / len(train_loader), train_acc_epoch / (len(train_loader) * train_loader.batch_size)

def test_epoch(model:nn.Module, device:str, test_loader:DataLoader, criterion):
    """
    Evaluates the model on the test dataset for one epoch.

    Args:
        model (nn.Module): The model to be evaluated.
        device (str): The device to be used for evaluation ("cuda" or "cpu").
        test_loader (DataLoader): The data loader for the test dataset.
        criterion: The loss function.

    Returns:
        float: The average test loss for the epoch.
        float: The average test accuracy for the epoch.
    """
    model.eval()
    test_loss_epoch = 0
    test_acc_epoch = 0
    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        test_loss_epoch += loss.item()
        test_acc_epoch += (y_pred.argmax(1) == y).sum().item()
    return test_loss_epoch / len(test_loader), test_acc_epoch / (len(test_loader) * test_loader.batch_size)


def train(model:nn.Module, device, train_loader, test_loader, criterion, optimizer, epochs, output_dir="models"):
    """
    Trains the model for the specified number of epochs.

    Args:
        model (nn.Module): The model to be trained.
        device (str): The device to be used for training ("cuda" or "cpu").
        train_loader (DataLoader): The data loader for the training dataset.
        test_loader (DataLoader): The data loader for the test dataset.
        criterion: The loss function.
        optimizer: The optimizer.
        epochs (int): The number of epochs to train the model.

    Returns:
        None
    """
    
    model.to(device)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    best_acc = 0
    best_acc_epoch = 0
    best_loss = float("inf")
    best_loss_epoch = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}:")
        train_loss_epoch, train_acc_epoch = train_epoch(model, device, train_loader, criterion, optimizer)
        print(f"Train loss: {train_loss_epoch}, Train accuracy: {train_acc_epoch}")
        test_loss_epoch, test_acc_epoch = test_epoch(model, device, test_loader, criterion)
        print(f"Val loss: {test_loss_epoch}, Val accuracy: {test_acc_epoch}")
        train_loss.append(train_loss_epoch)
        train_acc.append(train_acc_epoch)
        test_loss.append(test_loss_epoch)
        test_acc.append(test_acc_epoch)
        
        mlflow.log_metric("train_loss", train_loss_epoch, step=epoch)
        mlflow.log_metric("train_acc", train_acc_epoch, step=epoch)
        mlflow.log_metric("val_loss", test_loss_epoch, step=epoch)
        mlflow.log_metric("val_acc", test_acc_epoch, step=epoch)
        
        if test_acc_epoch > best_acc:
            torch.save(model.state_dict(), f"{output_dir}/best_acc_model.pth")
            best_acc = test_acc_epoch
            best_acc_epoch = epoch + 1
            
        if test_loss_epoch < best_loss:
            torch.save(model.state_dict(), f"{output_dir}/best_loss_model.pth")
            best_loss = test_loss_epoch
            best_loss_epoch = epoch + 1
        general_stats = pd.DataFrame({
            "best_acc": [best_acc],
            "best_acc_epoch": [best_acc_epoch],
            "best_loss": [best_loss],
            "best_loss_epoch": [best_loss_epoch]
        })
        general_stats.to_csv(f"{output_dir}/general_stats.csv")
        
def calculate_validation_accuracy(model:nn.Module, device, test_loader:DataLoader):
    """
    Calculates the validation accuracy for a trained model.

    Args:
        model (nn.Module): The trained model.
        device (str): The device to be used for prediction ("cuda" or "cpu").
        test_loader (DataLoader): The data loader for the test dataset.

    Returns:
        float: The validation accuracy.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            correct += (y_pred.argmax(1) == y).sum().item()
            total += len(y)
    return correct / total

def predict_one(model:nn.Module, device, x):
    """
    Makes a prediction for a single input using the trained model.

    Args:
        model (nn.Module): The trained model.
        device (str): The device to be used for prediction ("cuda" or "cpu").
        x: The input data.

    Returns:
        int: The predicted label.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        y_pred = model(x.unsqueeze(0))
        return y_pred.argmax(1).item()

##################################################################################################
##################################################################################################
if __name__ == "__main__":
    
    experiment_id = create_mlflow_experiment(
        experiment_name="MNIST",
        artifact_location="data/mlflow_artifacts",
        tags={"env": "Dev", "framework": "PyTorch", "version": "1.0.0"}
        )
    #mlflow.set_tracking_uri("http://localhost:5000")
    
    with mlflow.start_run(experiment_id=experiment_id):
        
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        params = {
            "batch_size": 32,
            "shuffle": True,
            "epochs": 10,
            "learningRate": 0.0001,
            "device": device
        }
        
        mlflow.log_params(params)
        
        
        random_num = r.randint(0, 10000)
        test_image, test_label = test_dataset[random_num]

        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=params["shuffle"], num_workers=4 if device == "cuda" else 0, persistent_workers=True if device == "cuda" else False)
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=2 if device == "cuda" else 0, persistent_workers=True if device == "cuda" else False)

        model = LeNet5()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learningRate"])

        with tempfile.TemporaryDirectory() as tmpdir:
            
            train(model, device, train_loader, test_loader, criterion, optimizer, params["epochs"], output_dir=tmpdir)
            mlflow.log_artifact(f"{tmpdir}/best_acc_model.pth", artifact_path="models")
            mlflow.log_artifact(f"{tmpdir}/best_loss_model.pth", artifact_path="models")
            mlflow.log_artifact(f"{tmpdir}/general_stats.csv", artifact_path="results")
            
            random_prediction = plt.figure()
            plt.imshow(test_image[0], cmap="gray")
            plt.title(f"True label: {test_label}, Predicted label: {predict_one(model, device, test_image)}")
            mlflow.log_figure(random_prediction, "random_prediction.png")
