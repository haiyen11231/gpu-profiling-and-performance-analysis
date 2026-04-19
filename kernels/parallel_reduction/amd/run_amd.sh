#!/bin/bash

hipcc reduction_amd.cpp -o reduction_amd
./reduction_amd > reduction_amd.log 2>&1
