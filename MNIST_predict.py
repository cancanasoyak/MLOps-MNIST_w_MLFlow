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
import matplotlib.pyplot as plt

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
        if self.num_workers == 0:
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        else:
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
    
    with mlflow.start_run(experiment_id=experiment_id, run_name="Prediction over random") as run:
        params = {
            "num_images": 20,
            "model_run_id": "d5f68d9a527e4bcdb8bf050cfdf90227"
        }
        
        mlflow.log_params(params)
        
        images = ImageFolder("data/testing", transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor()
            ]))
        
        # Create the model
        model = mlflow.pytorch.load_model(f"runs:/{params['model_run_id']}/model")
        
        mlflow.pytorch.autolog()
        # Create the trainer

        
        rndm_numbers = torch.randint(0, 10000, (params["num_images"],))
        #select images and put them into the same figure
        ncols = min(params["num_images"], 5)
        nrows = (params["num_images"] + ncols - 1) // ncols
        fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        for i, idx in enumerate(rndm_numbers):
            img = images[idx][0]
            label = images[idx][1]
            prediction = model(img.unsqueeze(0)).argmax().item()
            ax.ravel()[i].imshow(img.squeeze(), cmap='gray')
            ax.ravel()[i].set_title(f"Label: {label}, prediction: {prediction}")
            ax.ravel()[i].axis('off')

        mlflow.log_figure(fig, "predictions/random_predictions.png")

if __name__ == "__main__":
    main()

