## Usage

Data links are provided in appendix. Please download and unzip it under `src` directory. For example, `src/Brain10/Brian10.npz` should a valid path. The required packages are listed in `requirements.txt`, you can install them by running `pip install -r requirements.txt`.

Toy example is provided in a Jupiter notebook, which is under `examples` folder.

Running commands call to execution python file are under `examples` directory, you need to provide arguments to the script for learning rate and random seed you want to run. 
For example, for Brain10 dataset, `examples/task-train-brain10/brain10-none/Brain10~all~trans~16~softplus~1.0e-5~value~-1.sh` provide the script to run baseline methods, and `examples/task-train-brain10/brain10-none/IMP~Brain10~all~trans~none~16~softplus~1.0e-5~value~-1.sh` provide script to run our method. Please check the python code for the meaning of different component in the filename and possible arguments values, `~` is used to separate between different hyper-parameters.

Under `run_commands` folder, we provide example scripts demonstrate how to call to scripts inside `examples` command to run our experiments for corresponding dataset. You can modify hyperparameters by changing corresponding arguments in the bash command. If run successfully, it will create a pytorch saving file with file name includes all adjustable arguments you provide to the script with extension `.ptnnp` under `log` directory.

If you find our code useful, please consider citing our paper:

```
```