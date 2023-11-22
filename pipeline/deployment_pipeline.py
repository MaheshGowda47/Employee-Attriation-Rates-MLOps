import numpy as np
import pandas as pd
import json
import mlflow
import logging

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import ( MLFlowModelDeployer )
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output


from .utils import get_data_for_test
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.evaluate_data import evaluate_df
from steps.model_data import train_df

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
  """Class for configuring deployment trigger"""
  min_accuracy: float=0.0

@step
def deployment_trigger(
   accuracy: float,
   config: DeploymentTriggerConfig,
)-> bool:
   "Implement a single model deployment trigger that looks at the model accuracy and decide weather to deploy or not"
   return accuracy > config.min_accuracy

@pipeline(enable_cache=False, settings={"docker":docker_settings})
def continuous_deployment_pipeline(
   data_path: str,
   min_accuracy:float=0.0,
   workers: int=1,
   timeout: int=DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
   print("Deployment started .......")
   df=ingest_df(data_path)
   X_train,X_test,Y_train,Y_test=clean_df(df)
   model=train_df(X_train,X_test,Y_train,Y_test)
   evaluation_metrics=evaluate_df(model,X_test,Y_test)
   deployment_decision=deployment_trigger(evaluation_metrics)  
   print("deployment on go ..............")  
   mlflow_model_deployer_step(
      model=model,
      deploy_decision=deployment_decision,
      workers=workers,
      timeout=timeout,
    )
   print("deployment completed ..............")

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
   pipeline_name:str
   step_name:str
   running:bool=True


@step(enable_cache=False)
def dynamic_importer()->str:
   data=get_data_for_test()
   return data  


@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=21)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = ['BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently',
       'BusinessTravel_Travel_Rarely', 'Department_Human Resources',
       'Department_Research & Development', 'Department_Sales',
       'EducationField_Human Resources', 'EducationField_Life Sciences',
       'EducationField_Marketing', 'EducationField_Medical',
       'EducationField_Other', 'EducationField_Technical Degree',
       'Gender_Female', 'Gender_Male', 'JobRole_Healthcare Representative',
       'JobRole_Human Resources', 'JobRole_Laboratory Technician',
       'JobRole_Manager', 'JobRole_Manufacturing Director',
       'JobRole_Research Director', 'JobRole_Research Scientist',
       'JobRole_Sales Executive', 'JobRole_Sales Representative',
       'MaritalStatus_Divorced', 'MaritalStatus_Married',
       'MaritalStatus_Single', 'Age', 'DailyRate',
       'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
       'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',
       'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'OverTime', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager',]
    try:
      df = pd.DataFrame(data["data"], columns=columns_for_df)
      json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
      data = np.array(json_list)
      print("Input Data Shape:", data.shape)
      print("Input Data Sample:", data[:5])
      prediction = service.predict(data)
      return prediction
    except Exception as e:
        print(f"Prediction error: {str(e)}")


    
@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:

    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    logging.info(f"Existing Services: {existing_services}")
    print(f"Pipeline Name: {pipeline_name}, Step Name: {pipeline_step_name}, Model Name: {model_name}, Running: {running}")
    print(f"Existing Services: {existing_services}")

    if existing_services:  # Check if services exist
        return existing_services[0]  # Return the first service found
    else:
        raise RuntimeError("Failed to retrieve MLFlowDeploymentService")
   
@pipeline(enable_cache=False,settings={"docker":docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name:str):
   data=dynamic_importer()
   service=prediction_service_loader(
      pipeline_name=pipeline_name,
      pipeline_step_name=pipeline_step_name,
      running=False,
      )
   prediction=predictor(service=service,data=data)
   return prediction