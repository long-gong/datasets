#!/usr/bin/env bash

# Dataset dependency install script for Ubuntu and Fedora

# Fail on error
set -e 

# Echo on
set -x 

# Fail on unset var usage
set -o nounset

# Attempt to install command

DIST=Unknown

test -e /etc/debian_version && DIST="Debian"
grep Ubuntu /etc/lsb-release &> /dev/null && DIST="Ubuntu"
if [ "$DIST" = "Ubuntu" ] || [ "$DIST" = "Debian" ]; then
    # Truly non-interactive apt-get installation
    install='sudo DEBIAN_FRONTEND=noninteractive apt-get -y -q install'
    $install libhdf5-dev libeigen3-dev 
fi
test -e /etc/fedora-release && DIST="Fedora"
if [ "$DIST" = "Fedora" ]; then
    install='sudo dnf -y install'
    $install hdf5-devel eigen3-devel 
fi
test -e /etc/redhat-release && DIST="RedHatEnterpriseServer"
if [ "$DIST" = "RedHatEnterpriseServer" ]; then
    install='sudo dnf -y install'
    # TODO
fi
test -e /etc/SuSE-release && DIST="SUSE Linux"
if [ "$DIST" = "SUSE Linux" ]; then
    install='sudo zypper --non-interactive install '
    # TODO
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
