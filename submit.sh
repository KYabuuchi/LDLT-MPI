#!/bin/bash
array=(0)
for i in ${array[@]} ; do
    for j in {1..5} ; do
        FILE="run-${i}.sh"
        echo "pjsub ${FILE}"  
    done
done