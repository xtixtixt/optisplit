This is the accompanying code for the article 'Novel split quality measures for stratified multilabel cross validation with
application to large and sparse gene ontology datasets'. The experiments presented in the article can be
replicated by following the steps below:

1. Install dependencies

```
python3 -m venv .env && source .env/bin/activate && pip install -r requirements.txt
```

2. Run the experiments. (note that 10 experiments are run in parallel)

```
bash run_experiments.sh
```

3. Results will be generated to 'results' directory. The cross validation
method comparison results (Table 3 in the article) can be found in the files
'results/mean\_scores\*.csv'


An implementation of the optisplit algorithm and the new measures can be installed by running

```
pip install optisplit>=0.2
```

Usage example

```
import numpy as np
from optisplit import cv_balance
 
targets = np.random.binomial(1, 0.05,size=(1000,100))
folds = cv_balance.optisplit(targets=targets, n_splits=10)

rld_score = cv_balance.rld(folds=folds, targets=targets).mean()
```

