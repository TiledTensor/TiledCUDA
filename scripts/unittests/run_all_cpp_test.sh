#!/bin/bash

BUILD_DIR="build"

if [ ! -d $BUILD_DIR ]; then
    echo "This script should be run from the root of the project."
    exit 0
fi

cd $BUILD_DIR
# use ctest -R name_of_test to run a specific test
ctest
