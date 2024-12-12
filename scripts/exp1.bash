#!/bin/bash

# NOTE: run in yolov7/ folder

mkdir ./out
output="./out/run-$(date +%Y%m%d.%H%M.%S).out"
#output="./out/recent.out"
echo "$output"

declare -a weights=(
[0]=yolov7-tiny.pt
[1]=yolov7-x.pt
[2]=yolov7-w6.pt
[3]=yolov7-e6.pt
[4]=yolov7-d6.pt
[5]=yolov7-e6e.pt
)
declare -a batches=(
[0]=10
[1]=32
)

echo "weights:${weights[@]}" >> $output
echo "batches:${batches[@]}" >> $output

echo "...starting experiment one"

run_exp () {
    echo "...running exp $1 $2"
    echo "EXP $1 $2" >> $output
    perf stat -e power/energy-pkg/ python3 test.py --data data/coco.yaml --img 640 --batch $2 --conf 0.001 --iou 0.65 --weights $1 --name yolov7_640_val &>> $output
}

for i in "${weights[@]}"
do
    for j in "${batches[@]}"
    do
        run_exp "$i" "$j"
    done
done
