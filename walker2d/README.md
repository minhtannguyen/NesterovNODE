# Experiments with NODE-based models on Walker2D kinematic simulation task

## Download the data
The data can be downloaded by running
```
cd ..
python data_download.py
```

## Training
The code must be run from main directory of the repository, not from this directory. For example, if your working directory is ```walker2d```, first go up one level
```
cd ..
```
Then run the training process using this command:
```
python3 run.py walker <model-name> <choose your GPU number>
```
For example, if you want to run the experiment with NesterovNODE with GPU 0, then the command is as follows:
```
python3 run.py walker nesterovnode 0
```
Modify this command if you want to run with other models.

# Visualization
The code for plotting the results is in ```visualization/Walker2D_visualization.ipynb```. If the training process of the model is done successfully, follow the cells in this ```.ipynb``` file to reproduce the plots.