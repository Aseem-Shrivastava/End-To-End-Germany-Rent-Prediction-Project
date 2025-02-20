# Germany Apartment Rent Prediction Machine Learning Project

This is an end-to-end machine learning project designed to predict apartment rents in Germany. The model utilizes various data science techniques, including data exploration, feature engineering, model training, and deployment, to deliver accurate predictions of rental prices.

Key Features:
1.	In-depth Data Exploration & Analysis:
	- Comprehensive analysis of the dataset, uncovering key insights about apartment rent trends in Germany. 
	- Visualization of relationships between various features (e.g., location, size, amenities) and rent prices.
3.	Structured Data Processing:
	•	Implementation of robust data preprocessing techniques, including cleaning, encoding, and scaling, adhering to modular coding principles.
	•	Clean, well-documented code with reusable components for future scalability.
4.	MLOps Practices Integration:
	•	Process logging for better transparency and traceability of each step in the pipeline.
	•	Efficient training and testing pipelines for streamlined model evaluation.
	•	Experiment tracking and model management using MLflow, ensuring reproducibility and easier experimentation.
5.	CI/CD Pipeline with GitHub Actions:
	•	Automated continuous integration and deployment pipeline using GitHub Actions to ensure code quality and streamline the development process.
6.	Deployment on AWS EC2:
	•	Simple Flask application deployed on an AWS EC2 instance to serve the model for real-time predictions.
	•	Dockerized environment for easy scalability and seamless integration with cloud infrastructure.


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
├── Dockerfile                                      <- Docker file 
├── environment.yml                                 <- Environment file for reproducing the analysis environment
├── Makefile                                        <- Makefile with convenience commands like `make requirements`
├── pyproject.toml                                  <- Project configuration file with package metadata for 
│                                                   src and configuration for tools like black
├── README.md                                       <- The top-level README for developers using this project.
└── setup.cfg                                       <- Configuration file for flake8
```

--------
