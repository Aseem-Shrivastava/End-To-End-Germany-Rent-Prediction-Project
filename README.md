# Germany Apartment Rent Prediction

Machine learning model to predict apartment rents in Germany.

Highlights:
1. Thorough data exploration and insights
2. Structured data processing following modular coding principles
3. Integration of MLOps practices (Process logging, training and testing pipelines, experiment tracking with MLflow)
4. Deployment using simple Flask application


## Project Organization

```
├── analysis           
│   ├── EDA.ipynb                                   <- Exploratory data analysis (modular approach)
│   └── analysis_src                                <- Source code for packages used during exploratory data analysis
│       ├── __init__.py 
│       ├── basic_data_inspection.py
│       ├── bivariate_analysis.py
│       ├── missing_values_analysis.py
│       ├── multivariate_analysis.py
│       └── univariate_analysis.py
│
├── artifacts                   
│   └── preprocessor.pkl                            <- Preporcessor used for data transformation for training and test dataset
│
├── data
│   ├── extracted                                   <- Data extracted from raw data files (incase of zipped raw data)
│   └── raw                                         <- The original, immutable data dump
│       └── immo_data.csv
│
├── logs                                            <- Process logs for model training and predictions
│
├── models                                          
│   └── best_tuned_model.pkl                        <- Hyperparameter-tuned model used for prediction
│
├── src                                             <- Source code for use in this project (modular approach)
│   │
│   ├── components                                  <- Code for packages used during model training
│   │   ├── __init__.py 
│   │   ├── data_ingestion.py                
│   │   ├── data_splitter.py 
│   │   ├── data_transformation.py 
│   │   └── model_trainer.py
│   │
│   ├── pipeline                
│   │   ├── __init__.py 
│   │   ├── predict_pipeline.py                     <- Code to run model inference with trained models          
│   │   └── train_pipeline.py                       <- Code to test classification models and hyperparameter tune best fitted model
│   │
│   ├── __init__.py                                 <- Makes src a Python module
│   ├── config.py                                   <- Store useful variables and configuration
│   ├── logging_config.py                           <- Store useful variables and configuration for logging
│   └── utils.py                                    <- Scripts to save or load objects (e.g., preprocessor, trained model)
│
├── templates                                       <- HTML templates for FLASK application
│   ├── home.html                           
│   └── index.html                                    
│
├── app.py                                          <- Simple Flask application that allows users to predict apartment rent for custom data
├── environment.yml                                 <- Environment file for reproducing the analysis environment
├── Makefile                                        <- Makefile with convenience commands like `make requirements`
├── pyproject.toml                                  <- Project configuration file with package metadata for 
│                                                   src and configuration for tools like black
├── README.md                                       <- The top-level README for developers using this project.
└── setup.cfg                                       <- Configuration file for flake8
```

--------