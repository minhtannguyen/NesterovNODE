# Experiments with NODE-based models on Point Cloud benchmark

## Training
The code must be run from main directory of the repository, not from this directory. For example, if your working directory is ```point_cloud```, first go up one level
```
cd ..
```
Then run the training process using this command:
```
python3 nested_n_spheres.py --gpu <gpu> --output-directory point_cloud --names <model-name>
```
For example, if you want to run the experiment with NesterovNODE with GPU 0, then the command is as follows:
```
python3 nested_n_spheres.py --gpu 0 --output-directory point_cloud --names nesterovnode
```
Modify this command if you want to run with other models.

# Visualization
If the training process of the model is done successfully, use the following command to visualize the training results.

```
python3 nested_n_spheres.py --gpu <choose your GPU> --visualize-results 1 --output-directory point_cloud --names node anode sonode hbnode ghbnode nesterovnode gnesterovnode
```
You can additionally get the training visualization results by running:
```
python3 nested_n_spheres.py --gpu <choose your GPU> --visualize-features --num-epochs 100 --output-directory point_cloud_small --names node anode sonode hbnode ghbnode hdannode g5hdannode 
```