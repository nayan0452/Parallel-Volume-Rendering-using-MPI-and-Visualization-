#!/bin/bash

# Update the package list
sudo apt update

# Install Python3 and pip if not already installed
sudo apt install -y python3 python3-pip

# Install MPI (MPICH)
sudo apt install -y mpich

# Install required Python packages
pip3 install mpi4py numpy matplotlib
