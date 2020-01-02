#!/usr/bin/env bash

sudo apt-get install libhdf5-dev libeigen3-dev gio

cd /tmp
rm -rf HighFive xxHash
# shellcheck disable=SC2164
cd /tmp
git clone https://github.com/BlueBrain/HighFive.git
cd HighFive
mkdir build && cd build
cmake ..
make && sudo make install

cd /tmp
git clone https://github.com/Cyan4973/xxHash.git
cd xxHash
make && sudo make install

cd /tmp
rm -rf HighFive xxHash