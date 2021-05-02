python3 -m pip install virtualenv
virtualenv env
source env/bin/activate
sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
python3 -m pip install keras_applications==1.0.8 --no-deps
python3 -m pip install keras_preprocessing==1.1.0 --no-deps
python3 -m pip install h5py==2.9.0
sudo apt-get install -y openmpi-bin libopenmpi-dev
sudo apt-get install -y libatlas-base-dev
python3 -m pip install -U six wheel mock
wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.0.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl
python3 -m pip uninstall tensorflow
python3 -m pip install tensorflow-2.4.0-cp37-none-linux_armv7l.whl