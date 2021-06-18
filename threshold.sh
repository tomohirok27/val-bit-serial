#!/bin/bash


for i in `seq 8`
do
    python3 threshold.py -dn 0 -sn $i -tt 0.99 -pc True -pe True -ds pollux01  
done