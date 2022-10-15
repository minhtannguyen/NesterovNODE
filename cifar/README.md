# Experiments with NODE-based models on CIFAR10

## Training
The code must be run from main directory of the repository, not from this directory. For example, if your working directory is ```cifar```, first go up one level
```
cd ..
```
Then run the training process using this command:
```
python cifar/main.py --tol 1e-5  --gpu <choose your GPU number> --batch-size 256 --model <model-name>
```
For example, if you want to run the experiment with NesterovNODE with GPU 0, then the command is as follows:
```
python cifar/main.py --tol 1e-5  --gpu 0 --batch-size 256 --model nesterovnode
```
Modify this command if you want to run with other models.

# Visualization
The code for plotting the results is in ```visualization/CIFAR_visualization.py```. If the training process of the model is done successfully, run the following command:
```
python visualization/CIFAR_visualization.py
```