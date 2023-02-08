cwlVersion: v1.0

$graph:
- class: Workflow
  id: main
  inputs:
    model_directory: 
      type: Directory
      # default: /workspaces/mlflow-experiment/examples/sklearn_elasticnet_wine/mlruns/0
    model_id:
      type: string
    s2_data: 
      type: Directory
  outputs:
    model: 
      outputSource: node_infer/result
      type: File
  steps:
    node_infer:
      in: 
        model_directory: model_directory
        model_id: model_id
        s2_data: s2_data
      out: 
      - result
      run:
        "#infer"

- class: CommandLineTool
  id: infer
  requirements:
    InlineJavascriptRequirement: {}
    InitialWorkDirRequirement:
      listing:        
        - entryname: infer.py
          entry: |-
            import os
            import mlflow
            import sys
            import pandas as pd
            from skimage import io
            import numpy as np

            logged_model = sys.argv[1]
            img_folder = sys.argv[2]
            coll = io.ImageCollection(os.path.join(img_folder, "*.tif"))

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
  hints:
    DockerRequirement: 
      dockerPull: infer:latest
  baseCommand: ["python", "infer.py"]
  arguments: 
  - ${ return inputs.model_directory.path + "/" + inputs.model_id + "/artifacts/model" }
  - ${ return inputs.s2_data}
  inputs:
    s2_data:
      type: Directory
    model_directory: 
      type: Directory
    model_id: 
      type: string
  outputs:
    result: 
      type: File
      outputBinding:
        glob: classified.tif
