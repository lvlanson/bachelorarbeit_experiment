#!/bin/bash

datasets=("UCRArchive_2018/HouseTwenty/HouseTwenty_TRAIN.tsv,UCRArchive_2018/HouseTwenty/HouseTwenty_TEST.tsv" 
          "UCRArchive_2018/Beef/Beef_TRAIN.tsv,UCRArchive_2018/Beef/Beef_TEST.tsv" 
          "UCRArchive_2018/Wafer/Wafer_TRAIN.tsv,UCRArchive_2018/Wafer/Wafer_TEST.tsv")

prototype_count=(1 4 9)
epochs=(25 50 100)
distance_types=("fastdtw" "linmdtw")
lvq_types=("lvq1" "glvq")
gpu=(0 1)
exp_count=10
set=0

if test -f "protocol.yml";
then
  echo removing protocol
  rm "protocol.yml"
fi

for data in ${datasets[@]};
do
  path_train=${data#*","}
  path_test=${data%","*}
  
  for epoch in ${epochs[@]};
  do
    for prototype in ${prototype_count[@]};
    do
      for distance in ${distance_types[@]};
      do
        for lvq_type in ${lvq_types[@]};
        do
          echo Experiment $set started
          if [ "$distance" = "linmdtw" ]; then
            for g in ${gpu[@]};
            do
              for seed in $(seq $exp_count); 
              do
                # Mit GPU ON/OFF bei linmdtw
                python3 -W ignore Experiments.py $path_train $path_test $distance $lvq_type $prototype $epoch $g $seed $set
              done  
            done
          else
            for seed in $(seq $exp_count);
            do
              # GPU OFF bei fastDTW
              python3 -W ignore Experiments.py $path_train $path_test $distance $lvq_type $prototype $epoch 0 $seed $set
            done  
          fi
          let "set++"
        done
      done
    done
  done
done
