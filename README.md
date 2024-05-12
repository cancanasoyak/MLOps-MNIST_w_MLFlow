# MLOps-MNIST_w_MLFlow

first unzip the data.zip file, the folder structure should be:
    - ./data/testing
    - ./data/training


then to access the UI run:

    - mlflow ui

There are 3 main codes:

 - MNIST_lightning.py   (uses PyTorch Lightning to autolog)
 - MNIST_train.py       (manual logging)
 - MNIST_train_hp.py    (manual logging with hyperparameter search)

Just change the dictionaries named "params" and variable "run_name for mlflow,
you are ready to go.




jupyter notebook was just a testing environment