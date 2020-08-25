#!/usr/bin/env bash

# Dataset dependency install script for Ubuntu and Fedora

# Fail on error
set -e 

# Echo on
set -x 

# Fail on unset var usage
set -o nounset

if [ $TRAVIS_OS_NAME = 'osx' ]; then

    # Install some custom requirements on macOS
    # TODO
else
    # Install some custom requirements on Linux
    sudo apt-get install libhdf5-dev libeigen3-dev 
fi


cd /tmp
rm -rf HighFive xxHash
# shellcheck disable=SC2164
cd /tmp
git clone https://github.com/BlueBrain/HighFive.git
cd HighFive
mkdir build && cd build
cmake .. -DUSE_BOOST=FALSE
make && sudo make install

cd /tmp
git clone https://github.com/Cyan4973/xxHash.git
cd xxHash
make && sudo make install

cd /tmp
rm -rf HighFive xxHash
