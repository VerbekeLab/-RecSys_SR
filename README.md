# Data-driven internal mobility: Similarity regularization gets the job done </br><sub><sub>Simon De Vos, Johannes Desmedt, Marijke Verbruggen, Wouter Verbeke [[2024]](https://doi.org/10.1016/j.knosys.2024.111824)</sub></sub>
This paper presents a novel approach to support career management by recommending job–employee matches within an organization through data-driven insights. We build a recommender system to propose matches. The presented approach extends upon a conventional collaborative filtering recommender system, which suggests matches based on the historic performance similarities of employees. To address the prevalent challenge of the cold start issue in internal placement, we incorporate personal employee data with a similarity-based regularization term. This regularization term finds latent representations that are closer to each other when employees share similar personal features. This approach is evaluated using three real-life datasets and demonstrates a highly competitive performance compared to state-of-the-art benchmark methods. Overall, we make three contributions to the field of HR analytics: (i) we present a comprehensive survey of job–employee recommender systems in the context of internal mobility, (ii) we implement a similarity regularization method, and (iii) we release a first-of-its-kind HR dataset on internal mobility. 

## Instructions:
To replicate the results as reported in the paper, exectute the notebook *'main.ipynb'*

- Instructions for ['main.ipynb'](https://github.com/SimonDeVos/RecSys_SR/blob/master/experiment/main.ipynb):
  - Set the project directory to your custom folder. For example, `DIR = r'C:\Users\folder\subfolder\RecSys_SR'`
  - Specify experiment configuration in `settings = {'dataset': 3, 'method': 'mfsr', ...}`. The settings as currently specified will replicate the results on dataset 3. Per key, the following options are available:
    - `dataset: {3}`, 1 and 2 are not publicly available. Select 0 for a small synthetic dataset for quick runs and debugging.
    - `method: {'mf', 'mfsr', 'svd', 'slopeone', 'knnc', 'knnp', 'knnsmd', 'knnta'}`,
    - `test_size: [0,1]`,
    - `oot: {0,1}`, out-of-time split (no/yes)
    - `oitv: {0,1}`, ensures one observation per entity in the training set (no/yes)
    - `load_similarity: {False, True}`, If True, load similarity matrix from file, otherwise compute it. Relevant for methods: '_mfsr_', '_knnsmd_', '_knnta_'
    - `hyperpara_tune: {False, True}`, If True, tune hyperparameters. If False, load optimal hyperparameters from ['optimal_hyperparameters.yaml'](https://github.com/SimonDeVos/RecSys_SR/blob/master/experiment/optimal_hyperparameters.yaml)
  - Datasets 1 and 2 are not publicly available. Dataset 3 can be found [here](https://github.com/SimonDeVos/Anonymous_HR_event_log/blob/main/HR_log_anonymous.csv). 
  - Hyperparameter grids can be adapted in ['hyperparameter_grid.yaml'](https://github.com/SimonDeVos/RecSys_SR/blob/master/experiment/hyperparameter_grid.yaml)

## Project structure:
```
RecSys_SR/
│
├── data/
│   ├── toy_example.csv
│   └── dataset_3.csv
│
├── experiment/
│   ├── experiment.py: defines Experiment class and functions
│   ├── hyperparameter_grid.yaml: defines hyperparameter grid
│   ├── main.ipynb: run this to replicate experiments
│   ├── methods.py: defines methods ('mf', 'mfsr', 'svd', 'slopeone', 'knnc', 'knnp', 'knnsmd', 'knnta')
│   └── utils.py: contains utility functions
│
├── img/
│   ├── Two images on latent representations (Figures 2a and 2b).
│   └── EJM (Figure A.3)
│
└── similarity_matrix/
    └── stored_similarity_matrices for methods 'mfsr', 'knnsmd', 'knnta'
```

## Citing
Please cite our paper and/or code as follows:

```bash
De Vos, S., De Smedt, J., Verbruggen, M., & Verbeke, W. (2024). Data-driven internal mobility: Similarity regularization gets the job done. Knowledge-Based Systems, 111824.
```