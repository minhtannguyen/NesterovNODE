# Experiments with NODE-based models on MNIST

## Training
The code must be run from main directory of the repository, not from this directory. For example, if your working directory is ```mnist```, first go up one level
```
cd ..
```
Then run the training process using this command:
```
python mnist/mnist-full-run.py --gpu <choose your GPU number> --names <model-name> --log-file <model-log-file-name>
```
For example, if you want to run the experiment with NesterovNODE with GPU 0 and let the training log file name be nesterovnode, then the command is as follows:
```
python mnist/mnist-full-run.py --gpu 0 --names nesterovnode --log-file nesterovnode
```
Modify this command if you want to run with other models.

# Visualization
The code for plotting the results is in ```visualization/MNIST_visualization.ipynb```. If the training process of the model is done successfully, follow the cells in this ```.ipynb``` file to reproduce the plots.