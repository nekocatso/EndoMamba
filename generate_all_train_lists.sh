#!/bin/bash

# 为所有存在的数据集创建训练列表文件

datasets=(
    "/root/lanyun-tmp/EndoMamba-main/datasets/endoscopy/SUN-SEG"
    "/root/lanyun-tmp/EndoMamba-main/datasets/endoscopy/LDPolypVideo"
    "/root/lanyun-tmp/EndoMamba-main/datasets/endoscopy/Hyper-Kvasir"
    "/root/lanyun-tmp/EndoMamba-main/datasets/endoscopy/Kvasir-Capsule"
    "/root/lanyun-tmp/EndoMamba-main/datasets/endoscopy/EndoFM"
    "/root/lanyun-tmp/EndoMamba-main/datasets/endoscopy/GLENDA_v1.0"
    "/root/lanyun-tmp/EndoMamba-main/datasets/endoscopy/EndoMapper"
    "/root/lanyun-tmp/EndoMamba-main/datasets/surgery/CholecT45"
    "/root/lanyun-tmp/EndoMamba-main/datasets/surgery/ROBUST-MIS"
)

for dataset_dir in "${datasets[@]}"; do
    if [ -d "$dataset_dir" ]; then
        echo "Processing $dataset_dir"
        train_list="$dataset_dir/train_list.txt"
        
        # 清空或创建训练列表文件
        > "$train_list"
        
        # 查找所有视频文件并添加到训练列表
        find "$dataset_dir" -name "*.mp4" -o -name "*.avi" -o -name "*.mov" | while read video_file; do
            # 获取相对于数据集目录的路径
            relative_path=$(realpath --relative-to="$dataset_dir" "$video_file")
            echo "$relative_path 0" >> "$train_list"
        done
        
        echo "Created $train_list with $(wc -l < "$train_list") entries"
    else
        echo "Directory $dataset_dir does not exist, skipping"
    fi
done
