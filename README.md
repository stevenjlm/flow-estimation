# Network Flow Estimation
Estimating weights in a network given a time-series of data

## Dependencies

All the python packages requried to run the code will come with the Anaconda python platform.

## How to

### Run a Basic Demo

Will estimate 2 weights in a network considering the others as known. Runs in less than 30 minutes.

```bash
cd code
python 02Drv_GraphStruct.py
```

### Run a More Difficult Example

Will estimate 9 weights in a network. Does not always converge...

```bash
cd code
python 03_GibbsDebug.py
```

## Repository Organization

### The Code Directory

Contains all the python code for the project. The main scripts, or driver scripts are at the root. The Objs directory is a python package with all the modules that implement the project.

## Comments/Discussion

For the case of a linear ODE's I think Gibb's sampling is not a good method. A more straight-forward optimization approach is probably better and faster, so I will look into that more closely.
