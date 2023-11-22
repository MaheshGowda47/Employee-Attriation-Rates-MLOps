# Dataset: IBM HR Analytics Employee Attrition and Performance
[Source]: (https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)


**Age** -> age of particular employee [continues_numeric_data]

**Attrion** -> It is an target variable wheather the employee is in compamny or not [Target-variable] [Binary_ordinal_str_data] 

**BusinessTravel** -> Travel of employee for business perpose [multiclass_str_data]

**DailyRate** -> might represent the daily salary or pay rate of employees [continous_numeric_data]

**Department** -> department of an employee [multiclass_str_data]

**DistanceFromHome** -> Each value represents the distance from a location (presumably a workplace) to an employee's home [numeric_data]

**Education** -> It appears to represent some form of educational levels or categories.[numeric_data]

**EducationField** -> The education field of employee graducated. [multiclass_str_data]

**EmployeCount** -> Number of employee [not_important] [drop_column]

**EmployeeNumber** -> ordered number 1,2,3... [not_important] [drop_column]

**EvironmentSatisfaction** -> The values might represent categories like "Very Dissatisfied," "Dissatisfied," "Satisfied," and "Very Satisfied," [ordinal_numeric_data]

**Gender** -> Gender of employee [binary_str_data]

**HourlyRate** -> likely represents hourly wage rates for employees. [continues_numberic_data]

**JobInvolvement** -> likely represents different levels or categories of job involvement. For example, a value of "1" might represent the lowest or least level of job involvement, while "4" might represent the highest [multiclass_numeric_data]

**JobRole** -> which consists of job roles or job titles [multiclass_str_data]

**JobLevel** -> the job level of employee [multiclass_numeric_data] 

**JobSatisfaction** -> level of satisfaction in the job of an employe [multiclass_numeric_data]

**MaritalStatus** -> Marital status of an employee [multiclass_numeric_data]

**MonthlyIncome** -> monthly oncome of an employee [continous_numeric_data]

**MonthlyRate** -> like a dataset of monthly rates for some kind of financial or economic data [continous_numeric_data]

**NumCompaniesWorked** -> it represents the number of companies that individuals have worked for [continous_numeric_data]

**Over18** -> says the employee above 18 or not [not_important] [drop_column]

**OverTime** -> over time work of an employee [binary_str_data]

**PercentSalaryHike** -> percentage increase in salary for individuals [continous_numeric_data]

**PerformanceRating** -> rating for an employee based on performance [continous_numeric_data]

**RelationshipSatisfaction** -> Rating in his relationship [multiclass_numeric_data]

**StandardHours** -> standard hours working of an employee [not_important] [drop_colum]

**StockOptionLevel** -> stock option of an employee [multiclass_numeric_data]

**TotalWorkingYears** -> work experience of an employee [continous_numeric_data]

**TrainingTimesLastYear** -> training time of an employee last year [multivlass_numeric_data]

**WorkLifeBalance** -> how well individuals feel they can balance their work-related responsibilities with their personal or family life. [multiclass_numeric_data]

**YearsAtCompany** -> current working time in years of an employee [continous_numeric_data]

**YearsInCurrentRole** -> number working in the same role of an employee [continous_numeric_data]

**YearsSinceLastPromotion** -> number of working period in years after last promation [continous_numeric_data]

**YearsWithCurrManager** -> number of years spending with current manager [continous_numeric_data]





# ------------ Predicting Attrition Rate of an Employee in a company with ZenML and Streamlit ---------------- #


**problem atatement**: Our goal is to predict employee attrition in a company based on various factors such as income, age, performance, and personal details. To achieve this, we will leverage the IBM HR Analytics Employee Attrition & Performance dataset. We aim to create a production-ready pipeline using [ZenML](https://zenml.io/) to accurately forecast employee attrition. With [ZenML](https://zenml.io/) and Streamlit, we will build an end-to-end solution that empowers organizations to make informed decisions and proactively address attrition.

The purpose of this repository is to demonstrate how [ZenML](https://github.com/zenml-io/zenml) empowers your business to build and deploy machine learning pipelines in a multitude of ways:

- By offering you a framework and template to base your own work on.
- By integrating with tools like [MLflow](https://mlflow.org/) for deployment, tracking and more
- By allowing you to build and deploy your machine learning pipelines easily

## : Python Requirements

Let's jump into the Python packages you need, within the python environment of our choice, run:

```bash
git clone https://github.com/zenml-io/zenml-projects.git
cd zenml-Employee-Attrition-Rate-MLOps
pip install -r requirements.txt
```

Starting with ZenML 0.20.0, ZenML comes bundled with a React-based dashboard. This dashboard allows you to observe your stacks, stack components and pipeline DAGs in a dashboard interface. To access this, you need to [launch the ZenML Server and Dashboard locally](https://docs.zenml.io/user-guide/starter-guide#explore-the-dashboard), but first you must install the optional dependencies for the ZenML server:

```bash
pip install "zenml["server"]" # or
pip install zenml["server"]
zenml init                    #-to initialise the zenml reprository
zenml up                      #- to view dashboad 
```

If you are running the `run_deployment.py` or `run_pipeline.py` scripts, you should install some integrations using zenml:

```bash
zenml integration install mlflow -y
```
The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. Configuring a new stack with the two components are as follows:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker_employee --flavor=mlflow
zenml model-deployer register mlflow_employee --flavor=mlflow
zenml stack register mlflow_stack_employee -a default -o default -d mlflow_employee -e mlflow_tracker_employee --set

zenml stack describe

``` 

To view the experiment tracker in the browser, use the following:

```bash
mlflow ui --backend-store-uri "file:/root/.config/zenml/local_stores/1fb2f610-b8a4-4a4a-ba9b-0ec747164644/mlruns"
```

Install streamlit tor un our streamlit application
```bash
pip install streamlit
```
 

## :thumbsup: The Solution

In order to build a real-world workflow for predicting the attrition rate of employees (which will help to not loose the skilled employees), it is not enough to just train the model once.

Instead, we are building an end-to-end pipeline for continuously predicting and deploying the machine learning model, alongside a data application that utilizes the latest deployed model for the business to consume.

This pipeline can be deployed to the cloud, scale up according to our needs, and ensure that we track the parameters and data that flow through every pipeline that runs. It includes raw data input, features, results, the machine learning model and model parameters, and prediction outputs. ZenML helps us to build such a pipeline in a simple, yet powerful, way.

In this Project, we give special consideration to the [MLflow integration](https://github.com/zenml-io/zenml/tree/main/examples) of ZenML. In particular, we utilize MLflow tracking to track our metrics and parameters, and MLflow deployment to deploy our model. We also use [Streamlit](https://streamlit.io/) to showcase how this model will be used in a real-world setting.

### Training Pipeline

Our standard training pipeline consists of several steps:

- `ingest_data`: This step will ingest the data and create a `DataFrame`.
- `clean_data`: This step will clean the data and remove the unwanted columns.
- `train_model`: This step will train the model and save the model using [MLflow autologging](https://www.mlflow.org/docs/latest/tracking.html).
- `evaluation`: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact store.

### Deployment Pipeline

We have another pipeline, the `deployment_pipeline.py`, that extends the training pipeline, and implements a continuous deployment workflow. It ingests and processes input data, trains a model and then (re)deploys the prediction server that serves the model if it meets our evaluation criteria. The criteria that we have chosen is a configurable threshold on the [MSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) of the training. The first four steps of the pipeline are the same as above, but we have added the following additional ones:

- `deployment_trigger`: The step checks whether the newly trained model meets the criteria set for deployment.
- `model_deployer`: This step deploys the model as a service using MLflow (if deployment criteria is met).

In the deployment pipeline, ZenML's MLflow tracking integration is used for logging the hyperparameter values and the trained model itself and the model evaluation metrics -- as MLflow experiment tracking artifacts -- into the local MLflow backend. This pipeline also launches a local MLflow deployment server to serve the latest MLflow model if its accuracy is above a configured threshold.

The MLflow deployment server runs locally as a daemon process that will continue to run in the background after the example execution is complete. When a new pipeline is run which produces a model that passes the accuracy threshold validation, the pipeline automatically updates the currently running MLflow deployment server to serve the new model instead of the old one.

To round it off, we deploy a Streamlit application that consumes the latest model service asynchronously from the pipeline logic. This can be done easily with ZenML within the Streamlit code:

```python
service = prediction_service_loader(
   pipeline_name="continuous_deployment_pipeline",
   pipeline_step_name="mlflow_model_deployer_step",
   running=False,
)
...
service.predict(...)  # Predict on incoming data from the application
```

While this ZenML Project trains and deploys a model locally, other ZenML integrations such as the [Seldon](https://github.com/zenml-io/zenml/tree/main/examples/seldon_deployment) deployer can also be used in a similar manner to deploy the model in a more production setting (such as on a Kubernetes cluster). We use MLflow here for the convenience of its local deployment.

![training_and_deployment_pipeline](assets/training_and_deployment_pipeline_updated.png)

## :notebook: Diving into the code

You can run two pipelines as follows:

- Training pipeline:

```bash
python run_pipeline.py
```

- The continuous deployment pipeline:z

```bash
python run_deployment.py
```

To see the inference Pipeline,
```bash
   python run_deployment.py --config predict
   ```

## 🕹 Demo Streamlit App

There is a live demo of this project using [Streamlit](https://streamlit.io/) which you can find [here](https://share.streamlit.io/ayush714/customer-satisfaction/main). It takes some input features for the product and predicts the employee attrition rate using the latest trained models. If you want to run this Streamlit app in your local system, you can run the following command:-

```bash
streamlit run streamlit_app.py
```

## :question: FAQ

1. When running the continuous deployment pipeline, I get an error stating: `No Step found for the name mlflow_deployer`.

   Solution: It happens because your artifact store is overridden after running the continuous deployment pipeline. So, you need to delete the artifact store and rerun the pipeline. You can get the location of the artifact store by running the following command:

   ```bash
   zenml artifact-store describe
   ```

   and then you can delete the artifact store with the following command:

   **Note**: This is a dangerous / destructive command! Please enter your path carefully, otherwise it may delete other folders from your computer.

   ```bash
   rm -rf PATH
   ```

2. When running the continuous deployment pipeline, I get the following error: `No Environment component with name mlflow is currently registered.`

   Solution: You forgot to install the MLflow integration in your ZenML environment. So, you need to install the MLflow integration by running the following command:

   ```bash
   zenml integration install mlflow -y
   ```
3. If u see RuntimeError: Error initializing rest store with URL 'http://127.0.0.1:8237': HTTPConnectionPool(host='127.0.0.1', port=8237): Max retries exceeded with url:
/api/v1/info (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7fea7d7d5b70>: Failed to establish a new connection: [Errno 111] Connection
refused'))

     ```bash
         zenml down
         zenml up
         ```

