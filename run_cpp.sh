#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <scalar|vector> <O0|O1|O2|O3|Ofast> <demo|timing>"
    exit 1
fi

mode=$1
optimization=$2
demo_or_timing=$3

option="${mode^^}_${optimization}"

echo ${option}

if [ -d "build" ]; then
    echo "Removing existing build directory..."
    rm -rf build
fi
mkdir -p build
cd build

if [ ${demo_or_timing} = "timing" ]; then
    cmake -D${option}=ON -DLAB_TIMING=ON ../cpp
elif [ ${demo_or_timing} = "demo" ]; then
    cmake -D${option}=ON -DLAB_TIMING=OFF ../cpp
else
    echo "Must set demo or timing"
    exit
fi

make -j$(nproc)
