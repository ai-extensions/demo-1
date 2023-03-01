import sys
import os
import numpy as np

from skimage import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# mlflow
import mlflow
import mlflow.sklearn


def read_validation(validation_file, coll):
    p = np.loadtxt(validation_file, dtype="int")

    bands = []
    for i in range(len(coll)):
        band = np.asarray(coll[i].data)[p[:, 0], p[:, 1]]
        bands.append(band)

    return np.stack(bands).T


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":

    # model
    n_estimators = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    random_state = int(sys.argv[3])

    # read the labels and validation datasets
    training_water = sys.argv[4]
    training_artificial = sys.argv[5]
    training_low_vegetation = sys.argv[6]
    training_tree_cover = sys.argv[7]
    validation = sys.argv[8]
    image_folder = sys.argv[9]

    coll = io.ImageCollection(os.path.join(image_folder, "*.tif"))

    # Read the Sentinel-2 reflectances at the validation points
    X_water = read_validation(training_water, coll)
    X_artificial = read_validation(training_artificial, coll)
    X_low_veg = read_validation(training_low_vegetation, coll)
    X_trees = read_validation(training_tree_cover, coll)

    # stacking data for all classes in one table
    # the 400 training pixels for each land cover class and
    # 13 columns, reflecting the Sentinel-2 band information for each training point.
    X = np.vstack((X_water, X_artificial, X_low_veg, X_trees))

    # As output data (`y`), we create an array with the respective
    # land cover class of each training point.
    y = np.vstack(
        (
            np.ones((X_water.shape[0], 1)),
            2 * np.ones((X_artificial.shape[0], 1)),
            3 * np.ones((X_low_veg.shape[0], 1)),
            4 * np.ones((X_trees.shape[0], 1)),
        )
    )

    with mlflow.start_run(description="Random Forest classifier"):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        rf_clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

        # fit the model
        rf_clf.fit(X, np.ravel(y))

        # load the sentinel-2 data
        bands = []
        for i in range(len(coll)):
            band = np.asarray(coll[i].data).flatten()
            bands.append(band)

        X_all = np.stack(bands).T

        Y_pred_rf = rf_clf.predict(X_all)

        Y_im_rf = Y_pred_rf.reshape(coll[1].shape)

        # load validation dataset
        p_val = np.loadtxt(validation, dtype="int")

        pred_rf = Y_im_rf[p_val[:, 0], p_val[:, 1]]  # Random Forest

        (rmse, mae, r2) = eval_metrics(actual=p_val[:, 2], pred=pred_rf)
        acc_score = accuracy_score(p_val[:, 2], pred_rf)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("accuracy_score", acc_score)

        mlflow.sklearn.log_model(sk_model=rf_clf, artifact_path="model")
