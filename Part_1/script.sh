
# Update system package list
echo "Updating....."
sudo apt update -y

# Install MPICH
echo "Installing MPICH....."
sudo apt install -y mpich

# Install Python3 and pip3
echo "Installing Python3 and pip3....."
sudo apt install -y python3 python3-pip

# Install required Python packages
echo "Installing required Python packages: mpi4py, numpy, pillow, scipy....."
pip3 install --user mpi4py numpy pillow scipy

# Verify installations
echo "Verifying installations of all packages....."
mpichversion
python3 --version
pip3 freeze | grep -E 'mpi4py|numpy|pillow|scipy'

echo "Installation completed Now you can run volume_rendering.py file easily....."
