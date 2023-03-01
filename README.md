
# Demo-1 Sentinel-2 Random Forest Classifier - Supervised classification


This demo is based on the exercise [WEkEO ai4EM MOOC - Supervised classification using Sentinel-2 data](https://github.com/wekeo/ai4EM_MOOC/blob/main/3_land/3D2_land_cover_classification_with_Sentinel-2_data/3D2_supervised_classification_using_Sentinel-2.ipynb)

Use the Visual Studio development container to run this demo.

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

The inference takes the Sentinel-2 acquistion to classify with the provided model.

CWL can be used to do so with:

```
cwltool --no-container \
    infer.cwl \
    --infer infer.py \
    --model_directory mlruns/0/ \
    --model_id 593f7ddb798e49d0818e394d0b214b70 \
    --s2_data $PWD/S2_data/
```

Build the inference docker container with the selected model id:

```
docker build --build-arg model_id=593f7ddb798e49d0818e394d0b214b70 -f Dockerfile.infer .
```