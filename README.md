# DB_DiffCSP
Use DiffCSP to generate a new materila database

# DiffCSP

Implementation codes for Crystal Structure Prediction by Joint Equivariant Diffusion.

### Dependencies

```
python==3.8.13
torch==1.9.0
torch-geometric==1.7.2
pytorch_lightning==1.3.8
pymatgen==2022.9.21
```

### Training

```
python diffcsp/run.py data=<dataset> expname=<expname>
```

The ``<dataset>`` tag can be selected from perov_5, mp_20, mpts_52 and carbon_24.

### Evaluation

#### Stable structure prediction & Property prediction

One sample 

```
python scripts/evaluate.py --model_path <model_path>
python scripts/compute_metrics --root_path <model_path> --tasks struct --gt_file data/<dataset>/test.csv 
```

Multiple samples

```
python scripts/evaluate.py --model_path <model_path> --num_evals 20
python scripts/compute_metrics --root_path <model_path> --tasks struct prop --gt_file data/<dataset>/test.csv --multi_eval
```

#### Metastable structure generation

```
python scripts/generation.py --model_path <model_path> --dataset carbon
python scripts/compute_metrics --root_path <model_path> --tasks gen --gt_file data/carbon_24/test.csv
```


#### Sample from arbitrary composition

```
python scripts/sample.py --model_path <model_path> --save_path <save_path> --formula <formula> --num_evals <num_evals>
```
# DiffCSP_Default
