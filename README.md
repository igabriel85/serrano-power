# serrano-power
Repository for power consumption experiments

## Setup

Please cerate a virtual environment and install the requirements:

```bash
conda env create -f environment.yml
```

## Usage

For execution of experiments one requires first to create a YAML configuration file.
Curently, we support 4 detection methods:
- AdaBoost (see `ada.yaml`)
- Random Forest (see `rdf.yaml`)
- XGBoost (see `xgb.yaml`)
- LightGBM (see `lgb.yaml`)
- CatBoost (see `cat.yaml`)

### Configuration File
For each method, one can specify the following parameters:

- `method` - Name of method to be utilized, can be one of: `AdaBoostClassifier`, `RandomForestClassifier`, `XGBoost`, `LightGBM`, `CatBoost`

- `params` - Dictionary of parameters for the method, see example yaml files for examples and consul the documentation for each algorithm

- `datafraction` - Distribution of training data percentage to be used, each item from the distribution will result in a  newly trained predictive model. User has to define `start`, `end` and number of values to generate between `start` and `end`.

- `validationcurve` - List of parameters as well as value ranges to test. Each parameter value will undergo cross-validation as configured from the command line.

- `rfe` - Recurrent Feature elimination, by default starts with 1 feature and incrementally adds rest of features until complete. By default, cross-validation as defined by command line is used. User can define own list of features to be used, for example use ranking based on feature importance.

### Command line

  - -h, --help            show this help message and exit
  - -f FILE, --file FILE  configuration file (default: cat.yaml)
  - -d DATASET, --dataset DATASET
                        choose dataset, can be one of: anomaly, audsome, clean, clean_audsome (default: anomaly)
  - -i ITERATIONS, --iterations ITERATIONS
                        number of iterations (default: 1)
  - -cv CROSS_VALIDATION, --cross_validation CROSS_VALIDATION
                        number of splits used for cv (default: 5)
  - -cvt CROSS_VALIDATION_TEST, --cross_validation_test CROSS_VALIDATION_TEST
                        percent of test set (default: 0.2)
  - -e EXPERIMENT, --experiment EXPERIMENT
                        experiment unique identifier (default: exp_6)
  - -v, --verbose         verbosity (default: False)

##Example

- Run catboost example with anomaly dataset, one iteration and 5 fold Stratified Shuffle Split cross-validation, experiment id exp_1, verbose mode
```bash
python sr_energy_experiment.py -f cat.yaml -d anomaly -i 1 -cv 5 -cvt 0.2 -e exp_1 -v
```

- Run XGBoost example with audsome anomaly dataset, 10 iterations and 5 fold Stratified Shuffle Split cross-validation, experiment id exp_2, verbose mode
```bash
python sr_energy_experiment.py -f xgb.yaml -d audsome -i 10 -cv 5 -cvt 0.2 -e exp_2 -v
```

__Note__: Each experiment a new folder will be created with the name of the experiment id. In the model subfolder all reports and figures will be stored.