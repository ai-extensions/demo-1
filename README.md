
# Demo-1 Sentinel-2 Random Forest Classifier

## Training 

Using MLFlow:

```
mlflow run --env-manager local \
    -P training_water=./training_data/water.txt \
    -P training_artificial=./training_data/artificial_surfaces.txt \
    -P training_low_vegetation=./training_data/low_vegetation.txt \
    -P training_tree_cover=./training_data/tree_cover.txt \
    -P validation=./validation_data/validation_points.txt \
    -P img_folder=./S2_data
    .
```

Using a Common Workflow Language runner:

```
cwltool --no-container \
    train.cwl \
    --environment environment.yml \
    --train train.py \
    --ml_project MLproject \
    --max_depth 10 \
    --n_estimators 5 \
    --random_state 0 \
    --s2_data ./S2_data \
    --train_artificial_surfaces ./training_data/artificial_surfaces.txt \
    --train_low_vegetation ./training_data/low_vegetation.txt \
    --train_tree_cover ./training_data/tree_cover.txt \
    --train_water ./training_data/water.txt \
    --validation ./validation_data/validation_points.txt
```

## Inference

```
cwltool --no-container \
    infer.cwl \
    --model_directory mlruns/0/ \
    --model_id 847af2d92e0744baac9035f674e028d0 \
    --s2_data ./S2_data/
```