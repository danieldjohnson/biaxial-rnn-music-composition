sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy htop reptyr
sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git 
sudo wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb 
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb  
sudo apt-get update
sudo apt-get install -y cuda  
echo -e "\nexport PATH=/usr/local/cuda/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> .bashrc  
sudo reboot

sudo apt-get update
sudo apt-get -y dist-upgrade

screen -S “theano”
cuda-install-samples-7.0.sh ~/
cd NVIDIA_CUDA-7.0_Samples/
cd 1_Utilities/deviceQuery
make
./deviceQuery

cd /usr/bin/
sudo wget https://raw.githubusercontent.com/aurora/rmate/master/rmate
sudo chmod 775 rmate

# Mount new EBS volume (at sdf -> xvdf)
sudo fdisk /dev/xvdf
# Parameters:
# n
# p
# 1
#
#
# t
# 1
# 83
# w
sudo mkfs.ext3 -b 4096 /dev/xvdf

cd ~
mkdir external
sudo mount -t ext3 /dev/xvdf external/
sudo chmod 755 external
sudo chown ubuntu external
sudo chgrp ubuntu external
cd external
mkdir neural_music # and then get all files. Or maybe use git?

sudo dd if=/dev/zero of=~/external/swapfile1  bs=1024 count=4194304
sudo chown root:root ~/external/swapfile1
sudo chmod 0600 ~/external/swapfile1
sudo mkswap ~/external/swapfile1
sudo swapon ~/external/swapfile1

echo -e "\n[global]\nfloatX=float32\ndevice=gpu\nbase_compiledir=~/external/.theano/\nallow_gc=False\nwarn_float64=warn\n[mode]=FAST_RUN\n\n[nvcc]\nfastmath=True\n\n[cuda]\nroot=/usr/local/cuda\n" >> ~/.theanorc

sudo pip install theano-lstm

cd ~/external/neural_music
python
