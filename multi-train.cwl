cwlVersion: v1.0


$graph:
- class: Workflow
  id: main
  label: train several models
  requirements:
    SubworkflowFeatureRequirement: {}
    ScatterFeatureRequirement: {}
  inputs:
    environment: 
      type: File
    train:
      type: File
    ml_project:
      type: File
    n_estimators: 
      type: int[]
      default: 10
    max_depth:
      type: int[]
      default: 5
    random_state:
      type: int[]
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
      type: Directory[]
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
        "train.cwl"
      scatter: [n_estimators, max_depth, random_state]
      scatterMethod: flat_crossproduct