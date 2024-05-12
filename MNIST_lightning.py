import os

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
import mlflow.pytorch
from mlflow_utils import create_mlflow_experiment


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=4):
        """
        Initialization of inherited lightning data module
        """
        super().__init__()
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # transforms for images
        self.transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
        ])
        
        
    def setup(self, stage=None):
        """
        Loads the MNIST dataset and splits it into train, validation, and test sets.
        """
        
        self.dataset_train = ImageFolder("data/training", transform=self.transform)
        
        self.dataset_train, self.dataset_val = random_split(self.dataset_train, [55000, 5000])
        
        self.dataset_test = ImageFolder("data/testing", transform=self.transform)
        
    
    def create_data_loader(self, dataset, shuffle=False):
        """
        Create a DataLoader for the given dataset.
        
        :param dataset: The dataset to load.
        :return: DataLoader
        """
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, persistent_workers=True, num_workers=self.num_workers)
    
    def train_dataloader(self):
        """
        Create the DataLoader for the training set.
        
        :return: DataLoader
        """
        
        return self.create_data_loader(self.dataset_train, shuffle=True)
    
    def val_dataloader(self):
        """
        Create the DataLoader for the validation set.
        
        :return: DataLoader
        """
        
        return self.create_data_loader(self.dataset_val)
    
    def test_dataloader(self):
        """
        Create the DataLoader for the test set.
        
        :return: DataLoader
        """
        
        return self.create_data_loader(self.dataset_test)

class LightningMnistClassifier(L.LightningModule):
    def __init__(self, learning_rate=0.001):
        """
        Initialization of inherited lightning module
        """
        super().__init__()
        
        self.optimizer = None
        self.learning_rate = learning_rate
        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=0) 
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16*4*4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.loss_fnc = nn.CrossEntropyLoss()
        self.val_outputs = []
        self.test_outputs = []
        
    def forward(self, x):
        """
        Forward pass of the network
        """
            
        x = self.pool(self.relu(self.conv1(x))) # 1x28x28 -> 8x24x24 -> 8x12x12
        x = self.pool(self.relu(self.conv2(x))) # 8x12x12 -> 16x8x8 -> 16x4x4
        x = x.view(-1, 16*4*4) # 16x4x4 -> 256
        x = self.relu(self.fc1(x)) # 256 -> 64
        x = self.relu(self.fc2(x)) # 64 -> 32
        x = self.fc3(x) # 32 -> 10
        x = F.log_softmax(x, dim=1)
        return x
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        """
        
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fnc(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", accuracy(y_hat, y, task='multiclass', num_classes=10))
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step
        """
        
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fnc(y_hat, y)
        self.log("val_loss", loss)
        self.val_outputs.append((y_hat, y))
        self.log("val_acc", accuracy(y_hat, y, task='multiclass', num_classes=10))
        
    def test_step(self, batch, batch_idx):
        """
        Test step
        """
        
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fnc(y_hat, y)
        self.log("test_loss", loss)
        self.test_outputs.append((y_hat, y))
        self.log("test_acc", accuracy(y_hat, y, task='multiclass', num_classes=10))
    
    def configure_optimizers(self):
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return self.optimizer
        
        
    

def main():
    """
    Main function
    """
    experiment_id = create_mlflow_experiment(
        experiment_name="MNIST_lightning",
        artifact_location="data/mlflow_artifacts",
        tags={"env": "Dev", "framework": "PyTorch", "version": "1.0.0"}
        )
    #mlflow.set_tracking_uri("http://localhost:5000")
    
    with mlflow.start_run(experiment_id=experiment_id):
        params = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 20,
            "num_workers": 4
        }
        
        mlflow.log_params(params)
        
        # Create the data module
        data_module = MNISTDataModule(batch_size=params["batch_size"], num_workers=params["num_workers"])
        
        # Create the model
        model = LightningMnistClassifier(learning_rate=params["learning_rate"])
        
        mlflow.pytorch.autolog()
        # Create the trainer
        trainer = L.Trainer(
            max_epochs=params["num_epochs"],
            callbacks=[EarlyStopping(monitor='val_loss'), ModelCheckpoint(monitor='val_loss')],
            num_sanity_val_steps=0,
            log_every_n_steps=200
            )
        
        # Train the model
        trainer.fit(model, data_module)
        
        # Test the model
        trainer.test(model, data_module)
        
        # Save the model
        mlflow.pytorch.log_model(model, "mnist_model")
        

if __name__ == "__main__":
    main()