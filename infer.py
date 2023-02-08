import mlflow
import sys
import pandas as pd
from skimage import io
import numpy as np

logged_model = sys.argv[1]
img_folder = sys.argv[2]
coll = io.ImageCollection(img_folder + "*.tif")

bands = []
for i in range(len(coll)):
    band = np.asarray(coll[i].data).flatten()
    bands.append(band)

data = np.stack(bands).T

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
classified_data = loaded_model.predict(pd.DataFrame(data))

classified = classified_data.reshape(coll[1].shape)

io.imsave('classified.tif', classified.astype(np.uint8))