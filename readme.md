



python3 runner.py run.toml

python3 main.py load-rlivit -s 2 sink-mock handle-static

python3 converter.py rlivit2kitti -i ../../data/r-livit/R-LiViT_LiDAR/R-LiViT_LiDAR -o ../../data/r-livit-eval




python3 main.py load-rlivit -s 0 sink-ros2 handle-angle --epsilon1 0.1 --threshold 0.1 --min-samples1 4 --radius-smooth 0.4 --rm-ground --epsilon2 0.5 --min-samples2 4


python3 main.py load-rlivit -s 131 sink-ros2            handle-angle --threshold 0.2  --epsilon1 2 --min-samples1 10 --query-radius 0.4  --epsilon2 1 --min-samples2 10 --gate 5


python3 main.py load-ros2bag -i ../../data/lidar/02_indoor_pedestrian_movement/ sink-ros2 handle-beta -t 0.01 -r 0.1 -e1 0.2 -e2 0.5 -bcr 0.01 --gate 1


python3 main.py load-ros2bag -i ../../data/lidar/02_indoor_pedestrian_movement/ sink-ros2 handle-beta -t 0.01 -r 0.1 -e1 0.14 -e2 0.5 -bcr 0.01 --gate 1^C