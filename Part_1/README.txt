# MPI VOLUME RENDERING ASSIGNMENT README FILE

# Prerequisites :

Before running this code , ensure the following software is installed in your system:

1. MPI ( MPICH or openMPI )
2. PYTHON 3.x

And the following python packages are also required:

1. ‘mpi4py’
2. ‘numpy’
3. ‘pillow’
4. ‘scipy’

# Installation:

Step 1: Install MPI (MPICH or openMPI)
To install openMPI in Ubuntu, run:

sudo apt update 
sudo apt install -y openmpi-bin libopenmpi-dev

or To install MPICH in ubuntu, run:

sudo apt update
sudo apt install -y mpich

Step 2: Install Python 3.x:

sudo apt update
sudo apt install -y python3 python3-pip


Step 3: Install Python Packages, run:

pip3 install mpi4py numpy pillow scipy

## Or to install MPICH and all python packages you can simply run script.sh file:

chmod +x script.sh
./script.sh

# Running The Code:

After installing, you can run MPI based volume rendering program using the following command,

mpirun -np <num_processes> python3 volume_rendering.py <volume_file> <partition_method> <step_interval> <x_min> <x_max> <y_min> <y_max>

#Arguments:

<num_processes>: The number of processes
<volume_file>: The path to the .raw volume dataset file (e.g. Isabel_1000x1000x200_float32.raw).
<partition_method>: The partitioning method to divide the volume data (1 for 1Dimension or 2 for 2Dimension)
<step_interval>: The step size interval for ray casting (e.g. 0.5).
<x_min>, <x_max>: The bounds for the x-axis (e.g. 0  to 999).
<y_min>, <y_max>: The bounds for the y-axis (e.g. 0  to 999). 


