node_1=$(uniq $PBS_NODEFILE | head -n 1 | tail -n 1) 
node_2=$(uniq $PBS_NODEFILE | head -n 2 | tail -n 1) 
node_3=$(uniq $PBS_NODEFILE | head -n 3 | tail -n 1) 
node_4=$(uniq $PBS_NODEFILE | head -n 4 | tail -n 1) 
node_5=$(uniq $PBS_NODEFILE | head -n 5 | tail -n 1) 
node_6=$(uniq $PBS_NODEFILE | head -n 6 | tail -n 1) 
node_7=$(uniq $PBS_NODEFILE | head -n 7 | tail -n 1) 
node_8=$(uniq $PBS_NODEFILE | head -n 8 | tail -n 1) 
node_9=$(uniq $PBS_NODEFILE | head -n 9 | tail -n 1) 
node_10=$(uniq $PBS_NODEFILE | head -n 10 | tail -n 1) 
node_11=$(uniq $PBS_NODEFILE | head -n 11 | tail -n 1) 
node_12=$(uniq $PBS_NODEFILE | head -n 12 | tail -n 1) 
node_13=$(uniq $PBS_NODEFILE | head -n 13 | tail -n 1) 
node_14=$(uniq $PBS_NODEFILE | head -n 14 | tail -n 1) 
node_15=$(uniq $PBS_NODEFILE | head -n 15 | tail -n 1) 
node_16=$(uniq $PBS_NODEFILE | head -n 16 | tail -n 1) 
 
n_worker=$1 
n_cyc=$2 
case=$3 
n_rail=$4 
 
echo "config: " $n_worker $n_cyc $n_rail $case 
echo $(date) 
 
echo $node_1 
echo $node_2 
echo $node_3 
echo $node_4 
echo $node_5 
echo $node_6 
echo $node_7 
echo $node_8 
echo $node_9 
echo $node_10 
echo $node_11 
echo $node_12 
echo $node_13 
echo $node_14 
echo $node_15 
echo $node_16 
 
# 4 rail - theoretically best 
rail_send="mlx5_4:1,mlx5_8:1,mlx5_0:1,mlx5_2:1" 
rail_rec="mlx5_2:1,mlx5_0:1,mlx5_8:1,mlx5_4:1" 

./build_rankfile_hawk_ai.sh > myrf_16


mpirun --map-by rankfile:file=myrf_16 --tag-output --report-bindings \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \
 -n 6 ./bind.sh ./$case"_dsea" $n_worker $n_cyc $n_rail : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_2 `#sender` : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \
 -n 6 ./bind.sh ./$case"_dsea" $n_worker $n_cyc $n_rail : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_3 `#sender` : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \
 -n 6 ./bind.sh ./$case"_dsea" $n_worker $n_cyc $n_rail : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_4 `#sender` : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \
 -n 6 ./bind.sh ./$case"_dsea" $n_worker $n_cyc $n_rail : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_5 `#sender` : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \
 -n 6 ./bind.sh ./$case"_dsea" $n_worker $n_cyc $n_rail : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_6 `#sender` : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \
 -n 6 ./bind.sh ./$case"_dsea" $n_worker $n_cyc $n_rail : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_7 `#sender` : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \
 -n 6 ./bind.sh ./$case"_dsea" $n_worker $n_cyc $n_rail : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_8 `#sender` : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \
 -n 6 ./bind.sh ./$case"_dsea" $n_worker $n_cyc $n_rail : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_9 `#sender` : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \
 -n 6 ./bind.sh ./$case"_dsea" $n_worker $n_cyc $n_rail : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_10 `#sender` : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \
 -n 6 ./bind.sh ./$case"_dsea" $n_worker $n_cyc $n_rail : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_11 `#sender` : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \
 -n 6 ./bind.sh ./$case"_dsea" $n_worker $n_cyc $n_rail : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_12 `#sender` : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \
 -n 6 ./bind.sh ./$case"_dsea" $n_worker $n_cyc $n_rail : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_13 `#sender` : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \
 -n 6 ./bind.sh ./$case"_dsea" $n_worker $n_cyc $n_rail : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_14 `#sender` : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \
 -n 6 ./bind.sh ./$case"_dsea" $n_worker $n_cyc $n_rail : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_15 `#sender` : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \
 -n 6 ./bind.sh ./$case"_dsea" $n_worker $n_cyc $n_rail : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_16 `#sender` : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \
 -n 6 ./bind.sh ./$case"_dsea" $n_worker $n_cyc $n_rail : \
 -n 1 ./bind_ucx.sh ./$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_1 `#sender`

#mpirun --map-by rankfile:file=myrf_16 --tag-output --report-bindings \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \ 
#-n 6 ./bind.sh ./dsea_spin/$case"_dsea" $n_worker $n_cyc $n_rail : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_2 `#sender` : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \ 
#-n 6 ./bind.sh ./dsea_spin/$case"_dsea" $n_worker $n_cyc $n_rail : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_3 `#sender` : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \ 
#-n 6 ./bind.sh ./dsea_spin/$case"_dsea" $n_worker $n_cyc $n_rail : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_4 `#sender` : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \ 
#-n 6 ./bind.sh ./dsea_spin/$case"_dsea" $n_worker $n_cyc $n_rail : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_5 `#sender` : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \ 
#-n 6 ./bind.sh ./dsea_spin/$case"_dsea" $n_worker $n_cyc $n_rail : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_6 `#sender` : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \ 
#-n 6 ./bind.sh ./dsea_spin/$case"_dsea" $n_worker $n_cyc $n_rail : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_7 `#sender` : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \ 
#-n 6 ./bind.sh ./dsea_spin/$case"_dsea" $n_worker $n_cyc $n_rail : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_8 `#sender` : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \ 
#-n 6 ./bind.sh ./dsea_spin/$case"_dsea" $n_worker $n_cyc $n_rail : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_9 `#sender` : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \ 
#-n 6 ./bind.sh ./dsea_spin/$case"_dsea" $n_worker $n_cyc $n_rail : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_10 `#sender` : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \ 
#-n 6 ./bind.sh ./dsea_spin/$case"_dsea" $n_worker $n_cyc $n_rail : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_11 `#sender` : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \ 
#-n 6 ./bind.sh ./dsea_spin/$case"_dsea" $n_worker $n_cyc $n_rail : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_12 `#sender` : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \ 
#-n 6 ./bind.sh ./dsea_spin/$case"_dsea" $n_worker $n_cyc $n_rail : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_13 `#sender` : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \ 
#-n 6 ./bind.sh ./dsea_spin/$case"_dsea" $n_worker $n_cyc $n_rail : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_14 `#sender` : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \ 
#-n 6 ./bind.sh ./dsea_spin/$case"_dsea" $n_worker $n_cyc $n_rail : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_15 `#sender` : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \ 
#-n 6 ./bind.sh ./dsea_spin/$case"_dsea" $n_worker $n_cyc $n_rail : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_16 `#sender` : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_rec" $n_worker $n_cyc $n_rail -p 13337 -R $rail_rec : `#receiver` : \ 
#-n 6 ./bind.sh ./dsea_spin/$case"_dsea" $n_worker $n_cyc $n_rail : \ 
#-n 1 ./bind_ucx.sh ./dsea_spin/$case"_dsea_ucx_send" $n_worker $n_cyc $n_rail -p 13337 -R $rail_send -A $node_1 `#sender`