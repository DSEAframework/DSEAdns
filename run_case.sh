case=1024
supercycles=9
worker=1
rails=1


file=dsea_kernel_cycle00_base.cu
echo $file > $output

scp $file dsea_kernel.cu
./build_case.sh $case



timestamp=$(date +%F_%H-%M-%S)

output="./dsea_"$worker"_16_"$rails"_"$case"_"$supercycles"_"$timestamp".out"
./run_dsea_spin_16.sh $case $supercycles $worker > $output


