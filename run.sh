#!/bin/bash

/home/ubuntu/miniconda3/envs/pyml/bin/python sr_energy_experiments.py -f ada.yaml -e ada_exp
/home/ubuntu/miniconda3/envs/pyml/bin/python sr_energy_experiments.py -f rdf.yaml -e rdf_exp
/home/ubuntu/miniconda3/envs/pyml/bin/python sr_energy_experiments.py -f xgb.yaml -e xgb_exp
/home/ubuntu/miniconda3/envs/pyml/bin/python sr_energy_experiments.py -f lgbm.yaml -e lgbm_exp
/home/ubuntu/miniconda3/envs/pyml/bin/python sr_energy_experiments.py -f cat.yaml -e cat_exp