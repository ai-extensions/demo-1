cwlVersion: v1.0

$graph:
- class: Workflow
  id: main
  inputs:
    environment: 
      type: File
    train:
      type: File
    ml_project:
      type: File
    n_estimators: 
      type: int
      default: 10
    max_depth:
      type: int
      default: 5
    random_state:
      type: int
      default: 0
    train_artificial_surfaces: 
      type: File 
    train_low_vegetation: 
      type: File 
    train_tree_cover: 
      type: File
    train_water: 
      type: File 
    validation: 
      type: File
    s2_data: 
      type: Directory
  outputs:
    model: 
      outputSource: node_train/model
      type: Directory
  steps:
    node_train:
      in: 
        environment: environment
        train: train
        ml_project: ml_project
        n_estimators: n_estimators
        max_depth: max_depth
        random_state: random_state
        train_artificial_surfaces: train_artificial_surfaces
        train_low_vegetation: train_low_vegetation
        train_tree_cover: train_tree_cover
        train_water: train_water
        validation: validation
        s2_data: s2_data
      out: 
      - model
      run:
        "#train"

- class: CommandLineTool
  id: train
  requirements:
    InlineJavascriptRequirement: {}
    InitialWorkDirRequirement:
      listing:
      - $(inputs.environment)
      - $(inputs.train)
      - $(inputs.ml_project)
         
  hints:
    DockerRequirement: 
      dockerPull: train:latest 
  baseCommand: ["mlflow", "run", "--env-manager", "local"]
  arguments: 
  - ${ return "-Pn_estimators=" + inputs.n_estimators }
  - ${ return "-Pmax_depth=" + inputs.max_depth }
  - ${ return "-Prandom_state=" + inputs.random_state }
  - ${ return "-Ptraining_water=" + inputs.train_water.path }
  - ${ return "-Ptraining_artificial=" + inputs.train_artificial_surfaces.path }
  - ${ return "-Ptraining_low_vegetation=" + inputs.train_low_vegetation.path }
  - ${ return "-Ptraining_tree_cover=" + inputs.train_tree_cover.path }
  - ${ return "-Pvalidation=" + inputs.validation.path }
  - ${ return "-Pimg_folder=" + inputs.s2_data.path }
  - .
  inputs:
    environment:
      type: File
    train: 
      type: File
    ml_project: 
      type: File     
    n_estimators:
      type: int
    max_depth: 
      type: int
    random_state:
      type: int 
    train_artificial_surfaces: 
      type: File
    train_low_vegetation:
      type: File
    train_tree_cover:
      type: File
    train_water:
      type: File
    validation:
      type: File
    s2_data:
      type: Directory
  outputs:
    model: 
      type: Directory
      outputBinding:
        glob: mlruns
