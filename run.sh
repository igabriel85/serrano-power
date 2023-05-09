#!/bin/bash

/home/ubuntu/miniconda3/envs/pyml/bin/python sr_energy_experiments.py -f ada.yaml -e ada_exp_10 -cv 10
/home/ubuntu/miniconda3/envs/pyml/bin/python sr_energy_experiments.py -f rdf.yaml -e rdf_exp_10 -cv 10
/home/ubuntu/miniconda3/envs/pyml/bin/python sr_energy_experiments.py -f xgb.yaml -e xgb_exp_10 -cv 10
/home/ubuntu/miniconda3/envs/pyml/bin/python sr_energy_experiments.py -f lgbm.yaml -e lgbm_exp_10 -cv 10
/home/ubuntu/miniconda3/envs/pyml/bin/python sr_energy_experiments.py -f cat.yaml -e cat_exp_10 -cv 10