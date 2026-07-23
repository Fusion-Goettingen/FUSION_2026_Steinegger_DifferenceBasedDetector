# FUSION_2026_Steinegger_DifferenceBasedDetector
A detection and tracking pipeline on 3D point cloud based on frame-wise difference calculation


## Configuration 
mainly done over the run.toml

## Generating the results
```sh
python3 runner.py run.toml
```

### container

```sh
podman build -t pt:dev -f Containerfile.dev --no-cache
podman run -it -v ../pedestrian-tracker/data/:/data -v .:/app pt:dev
```

## Generating the plots
run the cells of the evaluation.jpynb notebook