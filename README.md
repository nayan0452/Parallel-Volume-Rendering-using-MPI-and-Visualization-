
# MPI Volume Rendering Project

This project implements **parallel volume rendering** using MPI for large 3D scalar datasets (e.g., Hurricane Isabel). 
It is divided into two parts, covering progressively advanced implementations of MPI-based ray casting and compositing.

---

## 📌 Part 1: MPI Volume Rendering (1D/2D Decomposition)

### Prerequisites
- MPI (MPICH or OpenMPI)
- Python 3.x
- Python packages:
  - mpi4py
  - numpy
  - pillow
  - scipy

### Installation
```bash
# Install MPI (choose one)
sudo apt update
sudo apt install -y openmpi-bin libopenmpi-dev   # OpenMPI
sudo apt install -y mpich                       # MPICH

# Install Python 3 and pip
sudo apt install -y python3 python3-pip

# Install required Python packages
pip3 install mpi4py numpy pillow scipy

# Or simply run the provided script
chmod +x script.sh
./script.sh
```

### Running the Code
```bash
mpirun -np <num_processes> python3 volume_rendering.py <volume_file> <partition_method> <step_interval> <x_min> <x_max> <y_min> <y_max>
```

**Arguments:**
- `<num_processes>`: Number of MPI processes  
- `<volume_file>`: Path to `.raw` volume dataset (e.g., Isabel_1000x1000x200_float32.raw)  
- `<partition_method>`: Partitioning method (1 = 1D, 2 = 2D)  
- `<step_interval>`: Step size interval for ray casting (e.g., 0.5)  
- `<x_min>, <x_max>`: Bounds for x-axis (e.g., 0–999)  
- `<y_min>, <y_max>`: Bounds for y-axis (e.g., 0–999)  

---

## 📌 Part 2: MPI Volume Rendering (3D Decomposition & Transfer Functions)

### Prerequisites
- MPI (MPICH or OpenMPI)
- Python 3.x
- Python packages:
  - mpi4py
  - numpy
  - matplotlib

### Installation
```bash
# Same installation steps as Part 1, with matplotlib instead of pillow/scipy
pip3 install mpi4py numpy matplotlib
```

### Running the Code
```bash
# Example: 8 processes
mpirun -np 8 python3 volume_rendering_3D.py Isabel_1000x1000x200_float32.raw 2 2 2 0.5 opacity_TF.txt color_TF.txt

# Example: 16 processes
mpirun -np 16 python3 volume_rendering_3D.py Isabel_1000x1000x200_float32.raw 2 2 4 0.5 opacity_TF.txt color_TF.txt

# Example: 32 processes
mpirun -np 32 python3 volume_rendering_3D.py Isabel_1000x1000x200_float32.raw 2 2 8 0.5 opacity_TF.txt color_TF.txt
```

### Running on Large Dataset
```bash
mpirun -np 8  python3 volume_rendering_3D.py Isabel_2000x2000x400_float32.raw 2 2 2 0.5 opacity_TF.txt color_TF.txt
mpirun -np 16 python3 volume_rendering_3D.py Isabel_2000x2000x400_float32.raw 2 2 4 0.5 opacity_TF.txt color_TF.txt
mpirun -np 32 python3 volume_rendering_3D.py Isabel_2000x2000x400_float32.raw 2 2 8 0.5 opacity_TF.txt color_TF.txt
```

### Running on Multiple Nodes
```bash
mpirun --mca btl_tcp_if_include eno1 --hostfile hostfile -np 8  python3 volume_rendering_3D.py Isabel_1000x1000x200_float32.raw 2 2 2 0.5 opacity_TF.txt color_TF.txt
mpirun --mca btl_tcp_if_include eno1 --hostfile hostfile -np 16 python3 volume_rendering_3D.py Isabel_1000x1000x200_float32.raw 2 2 4 0.5 opacity_TF.txt color_TF.txt
mpirun --mca btl_tcp_if_include eno1 --hostfile hostfile -np 32 python3 volume_rendering_3D.py Isabel_1000x1000x200_float32.raw 2 2 8 0.5 opacity_TF.txt color_TF.txt
```

**Notes:**
- Add `--oversubscribe` if processes exceed available cores.  
- If multi-node commands fail, adjust `--hostfile` syntax.  
- Contact for support: `deepaksoni24@iitk.ac.in`  

**Arguments:**
- `<num_processes>`: Number of MPI processes  
- `<volume_file>`: Path to `.raw` dataset (e.g., Isabel_1000x1000x200_float32.raw)  
- `<x_divs>, <y_divs>, <z_divs>`: Number of domain divisions along x, y, z axes  
- `<step_interval>`: Step size interval for ray casting (e.g., 0.5)  
- `<opacity_tf_file>`: Path to opacity transfer function file  
- `<color_tf_file>`: Path to color transfer function file  

---

## ⚙️ Technology Stack
Python, MPI (MPICH/OpenMPI), mpi4py, NumPy, Pillow, SciPy, Matplotlib  

---
