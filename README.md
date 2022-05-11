## Tools used in this project
* [Poetry](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f): Dependency management - [article](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f)
* [hydra](https://hydra.cc/): Manage configuration files - [article](https://towardsdatascience.com/introduction-to-hydra-cc-a-powerful-framework-to-configure-your-data-science-projects-ed65713a53c6)
* [Prefect](https://www.prefect.io): Orchestration - [article](https://towardsdatascience.com/orchestrate-a-data-science-project-in-python-with-prefect-e69c61a49074)
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting  - [article](https://towardsdatascience.com/4-pre-commit-plugins-to-automate-code-reviewing-and-formatting-in-python-c80c6d2e9f5?sk=2388804fb174d667ee5b680be22b8b1f)
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation for your project

## Project structure
```bash
.
├── config                      
│   ├── main.yaml                   # Main configuration file
│   ├── model                       # Configurations for training model
│   │   ├── model.yaml             # First variation of parameters to train model
├── data            
│   ├── processed                   # data after processing
│   ├── raw                         # data before pricessing
│   └── features_data               # data before modelling
├── docs                            # documentation
├── .flake8                         # configuration for flake8 - a Python formatter tool
├── .gitignore                      # ignore files that cannot commit to Git
├── Makefile                        # store useful commands to set up the environment
├── models                          # store models
├── notebooks                       # store notebooks
├── .pre-commit-config.yaml         # configurations for pre-commit
├── pyproject.toml                  # dependencies for poetry
├── README.md                       # describe your project
├── Dockerfile                      # Dockerfile
├── src                             # source code
│   ├── __init__.py                 # make src a Python module 
│   ├── process.py                  # process data before training model
│   └── feature_engineer.py         # create features
│   └── feature_main.py             # run process and feature engineer
│   └── train_model.py              # train model
└── tests                           # store tests
    ├── __init__.py                 # make tests a Python module 
    ├── test_process.py             # test functions for process.py
└── app                             # app for docker
    ├── main.py                     # FastAPI module with sklearn pipelines
    ├── serve.py                    
```

## Set up the environment
1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Set up the environment:
```bash
make install
make activate
```

3. To persist the output of Prefect's flow, run 
```bash
export PREFECT__FLOWS__CHECKPOINTING=true
```

## Run the Project
To run all flows, type:
```bash
python src/feature_main.py
python src/train_model.py
```

## Run Tests
To run all flows, type:
```bash
make test
```

## Run FastAPI through Docker
1. Build the Docker Image
```bash
docker build -t testliodocker ./  
```
2. Run the Docker Image
```bash
docker run -d --name testliodocker -p 80:80 testliodocker
```

3. Test the API-Go to http://127.0.0.1/docs#/default/predict_predict_post

