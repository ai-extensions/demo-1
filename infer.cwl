cwlVersion: v1.0

$graph:
- class: Workflow
  id: main
  inputs:
    infer: 
      type: File
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
        infer: infer
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
        - $(inputs.infer)        
        
  hints:
    DockerRequirement: 
      dockerPull: infer:latest
  baseCommand: ["python", "infer.py"]
  arguments: 
  - ${ return inputs.model_directory.path + "/" + inputs.model_id + "/artifacts/model" }
  - ${ return inputs.s2_data}
  inputs:
    infer:
      type: File
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
