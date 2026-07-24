# FUSION_2026_Steinegger_DifferenceBasedDetector
A detection and tracking pipeline on 3D point cloud based on frame-wise difference calculation

## Setup
Clone this repository.
Download the dataset.
And run the commands to recreate the plot of this paper

For evaluation the [R-LiViT](https://github.com/XITASO/r-livit) data set is used.
Download the dataset und put it into your folder of choice (Here ./data).

## Configuration 
Done over the run.toml
- set the path to your data
- control the setting of the method
    - The current parameters are the parameters used for the evaluation

## Generating the results
```sh
python3 runner.py run.toml
```

### container
```sh
podman build -t dt:dev -f Containerfile.dev --no-cache
podman run --rm -it -v ./data/:/data -v .:/app dt:dev
```

## Generating the plots
run the cells of the evaluation.jpynb notebook