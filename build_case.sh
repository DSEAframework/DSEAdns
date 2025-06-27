case=$1

echo $case
echo "#define case_"$case > dsea_case.h

make clean
make
./move_bin_to_case.sh $case

