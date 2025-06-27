
for case in 64 90 128 180 256 362 512 724 1024 1200 1448 1800 1850 1870
do
	echo $case
	echo "#define case_"$case > dsea_case.h

	make clean
	make
	./move_bin_to_case.sh $case
done	
