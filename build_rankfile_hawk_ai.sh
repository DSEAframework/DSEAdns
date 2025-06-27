#!/bin/bash
#echo $PBS_NODEFILE
nodelist=$(uniq $PBS_NODEFILE | sort)


irank=0

for node in $nodelist
do
#	echo $node
	echo rank $(expr $irank + 0)=$node slot=48-55
        echo rank $(expr $irank + 1)=$node slot=56-63
        echo rank $(expr $irank + 2)=$node slot=16-23
        echo rank $(expr $irank + 3)=$node slot=24-31
        echo rank $(expr $irank + 4)=$node slot=112-119
        echo rank $(expr $irank + 5)=$node slot=120-127
        echo rank $(expr $irank + 6)=$node slot=80-87
	echo rank $(expr $irank + 7)=$node slot=88-95
	irank=$(expr $irank + 8)
done

