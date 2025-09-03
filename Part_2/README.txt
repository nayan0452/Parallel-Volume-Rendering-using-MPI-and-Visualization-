# MPI VOLUME RENDERING ASSIGNMENT 2 README FILE

# Prerequisites :

Before running this code , ensure the following software is installed in your system:

1. MPI ( MPICH or openMPI )
2. PYTHON 3.x

And the following python packages are also required:

1. ‘mpi4py’
2. ‘numpy’
3. ‘matplotlib'

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

pip3 install mpi4py numpy matplotlib

## Or to install MPICH and all python packages you can simply run script.sh file:

chmod +x script.sh

./script.sh

# Running The Code:

After installing, you can run MPI based volume rendering program using the following command,

1. 

mpirun -np 8 python3 volume_rendering_3D.py Isabel_1000x1000x200_float32.raw 2 2 2 0.5 opacity_TF.txt color_TF.txt

Or

mpirun --oversubscribe -np 8 python3 volume_rendering_3D.py Isabel_1000x1000x200_float32.raw 2 2 2 0.5 opacity_TF.txt color_TF.txt

2.

mpirun -np 16 python3 volume_rendering_3D.py Isabel_1000x1000x200_float32.raw 2 2 4 0.5 opacity_TF.txt color_TF.txt

Or

mpirun --oversubscribe -np 16 python3 volume_rendering_3D.py Isabel_1000x1000x200_float32.raw 2 2 4 0.5 opacity_TF.txt color_TF.txt

3. 

mpirun -np 32 python3 volume_rendering_3D.py Isabel_1000x1000x200_float32.raw 2 2 8 0.5 opacity_TF.txt color_TF.txt

Or

mpirun --oversubscribe -np 32 python3 volume_rendering_3D.py Isabel_1000x1000x200_float32.raw 2 2 8 0.5 opacity_TF.txt color_TF.txt



# For Large Dataset :-

1.

mpirun -np 8 python3 volume_rendering_3D.py Isabel_2000x2000x400_float32.raw 2 2 2 0.5 opacity_TF.txt color_TF.txt

Or

mpirun --oversubscribe -np 8 python3 volume_rendering_3D.py Isabel_2000x2000x400_float32.raw 2 2 2 0.5 opacity_TF.txt color_TF.txt

2.

mpirun -np 16 python3 volume_rendering_3D.py Isabel_2000x2000x400_float32.raw 2 2 4 0.5 opacity_TF.txt color_TF.txt

Or

mpirun --oversubscribe -np 16 python3 volume_rendering_3D.py Isabel_2000x2000x400_float32.raw 2 2 4 0.5 opacity_TF.txt color_TF.txt

3.

mpirun -np 32 python3 volume_rendering_3D.py Isabel_2000x2000x400_float32.raw 2 2 8 0.5 opacity_TF.txt color_TF.txt

Or

mpirun --oversubscribe -np 32 python3 volume_rendering_3D.py Isabel_2000x2000x400_float32.raw 2 2 8 0.5 opacity_TF.txt color_TF.txt


# To run the code on Nodes :-

1.

mpirun --mca btl_tcp_if_include eno1 --hostfile hostfile -np 8 python3 volume_rendering_3D.py Isabel_1000x1000x200_float32.raw 2 2 2 0.5 opacity_TF.txt color_TF.txt

2.

mpirun --mca btl_tcp_if_include eno1 --hostfile hostfile -np 16 python3 volume_rendering_3D.py Isabel_1000x1000x200_float32.raw 2 2 4 0.5 opacity_TF.txt color_TF.txt

3.

mpirun --mca btl_tcp_if_include eno1 --hostfile hostfile -np 32 python3 volume_rendering_3D.py Isabel_1000x1000x200_float32.raw 2 2 8 0.5 opacity_TF.txt color_TF.txt



NOTE :- If above code is not running please add --oversubscribe

NOTE :- If multiple nodes command is not running please remove one - from ( --hostfile ) 

Note :- If you are having any issue in running the code please contact us at ( deepaksoni24@iitk.ac.in )


#Arguments:

<num_processes>: The number of processes
<volume_file>: The path to the .raw volume dataset file (e.g. Isabel_1000x1000x200_float32.raw).
<x_divs>, <y_divs>, <z_divs>: Number of divisions in the x, y, and z axes respectively.
<step_interval>: The step size interval for ray casting (e.g. 0.5).
<opacity_tf_file>: Path to the opacity transfer function file.
<color_tf_file>: Path to the color transfer function file.


