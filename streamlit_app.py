import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipeline.deployment_pipeline import prediction_service_loader
from run_deployment import run_deployment

# Define a global variable to keep track of the service status
service_started = False

def start_service():
    global service_started
    service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
    )
    service.start(timeout=21)  # Start the service
    service_started = True

    return service

def stop_service(service):
    global service_started
    service.stop()  # Stop the service
    service_started = False

def run_deployment():
    st.title("Employee Attrition Prediction")

    
    categorical_features = {
    "BusinessTravel": ('Travel_Frequently', 'Travel_Rarely', 'Non-Travel'),
    "Department": ('Sales', 'Research & Development', 'Human Resources'),
    "EducationField": ('Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'),
    "Gender": ('Female', 'Male'),
    "JobRole": ('Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare', 'Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'),
    "MaritalStatus": ('Single', 'Married', 'Divorced')
}
    for feature, options in categorical_features.items():
        categorical_features[feature] = st.sidebar.selectbox(feature, options)
    #below are numerical features
    age = st.sidebar.slider("Age", 18, 90, 30)
    daily_rate = st.sidebar.slider("Daily Rate", 0, 1500, 0)
    distance_from_home = st.sidebar.slider("Distance From Home", 0, 30, 0)
    education = st.sidebar.slider("Education", 1, 5, 1)
    environment_satisfaction = st.sidebar.slider("Environment Satisfaction", 1, 4, 1)
    hourly_rate = st.sidebar.slider("Hourly Rate", 0, 100, 0)
    job_involvement = st.sidebar.slider("Job Involvement", 1, 4, 1)
    job_level = st.sidebar.slider("Job Level", 1, 5, 1)
    job_satisfaction = st.sidebar.slider("Job Satisfaction", 1, 4, 1)
    monthly_income = st.sidebar.slider("Monthly Income", 0, 100000, 0)
    monthly_rate = st.sidebar.slider("Monthly Rate", 0, 30000, 0)
    num_comp_worked_prev = st.sidebar.slider("Number of Companies Worked Previously", 0, 10, 0)
    overtime=st.sidebar.selectbox('Overtime', ('Yes', 'No'))
    percent_salary_hike = st.sidebar.slider("Percent Salary Hike", 0, 30000, 0)
    performance_rating = st.sidebar.slider("Performance Rating", 1, 4, 1)
    relationship_satisfaction = st.sidebar.slider("Relationship Satisfaction", 1, 4, 1)
    stock_option_level = st.sidebar.slider("Stock Option Level", 1, 4, 1)
    total_working_years = st.sidebar.slider("Total Working Years", 0, 50, 0)
    training_times_last_year = st.sidebar.slider("Training Times Last Year", 0, 10, 0)
    work_life_balance = st.sidebar.slider("Work Life Balance", 1, 4, 1)
    years_at_company = st.sidebar.slider("Years at Company", 0, 50, 0)
    years_in_current_role = st.sidebar.slider("Years in Current Role", 0, 50, 0)
    years_since_last_promotion = st.sidebar.slider("Years Since Last Promotion", 0, 50, 0)
    years_with_current_manager = st.sidebar.slider("Years with Current Manager", 0, 50, 0)



    
    if st.button("Predict"):
        global service_started
        if not service_started:
            service = start_service()
        print("Prediction button clicked")

        input_data = {
            "BusinessTravel_Non-Travel": 0,
            "BusinessTravel_Travel_Frequently": 0,
            "BusinessTravel_Travel_Rarely": 0,
            "Department_Human Resources": 0,
            "Department_Research & Development": 0,
            "Department_Sales": 0,
            "EducationField_Human Resources": 0,
            "EducationField_Life Sciences": 0,
            "EducationField_Marketing": 0,
            "EducationField_Medical": 0,
            "EducationField_Other": 0,
            "EducationField_Technical Degree": 0,
            "Gender_Female": 0,
            "Gender_Male": 0,
            "JobRole_Healthcare Representative": 0,
            "JobRole_Human Resources": 0,
            "JobRole_Laboratory Technician": 0,
            "JobRole_Manager": 0,
            "JobRole_Manufacturing Director": 0,
            "JobRole_Research Director": 0,
            "JobRole_Research Scientist": 0,
            "JobRole_Sales Executive": 0,
            "JobRole_Sales Representative": 0,
            "MaritalStatus_Divorced": 0,
            "MaritalStatus_Married": 0,
            "MaritalStatus_Single": 0,
            'Age': int(age),
            'DailyRate': float(daily_rate),
            'DistanceFromHome': float(distance_from_home),
            'Education': int(education),
            'EnviromentSatisfaction': int(environment_satisfaction),
            'HourlyRate': float(hourly_rate),
            'JobInvolvement': int(job_involvement),
            'JobLevel': int(job_level),
            'JobSatisfaction': int(job_satisfaction),
            'MonthlyIncome': float(monthly_income),
            'MonthlyRate': float(monthly_rate),
            'NumCompaniesWorkedPrev': int(num_comp_worked_prev),  
            'Overtime': 1 if str(overtime) == 'Yes' else 0,  
            'PercentSalaryHike': float(percent_salary_hike),
            'PerformanceRating': int(performance_rating),
            'RelationshipSatisfaction': int(relationship_satisfaction),
            'StockOptionLevel': int(stock_option_level),
            'TotalWorkingYears': int(total_working_years),
            'TrainingTimesLastYear': int(training_times_last_year),
            'WorkLifeBalance': int(work_life_balance),
            'YearsAtCompany': int(years_at_company),
            'YearsInCurrentRole': int(years_in_current_role),
            'YearsSinceLastPromotion': int(years_since_last_promotion),
            'YearsWithCurrentManager': int(years_with_current_manager),
        }

        input_data[f"BusinessTravel_{categorical_features['BusinessTravel']}"] = 1
        input_data[f"Department_{categorical_features['Department']}"] = 1
        input_data[f"EducationField_{categorical_features['EducationField']}"] = 1
        input_data[f"Gender_{categorical_features['Gender']}"] = 1
        input_data[f"JobRole_{categorical_features['JobRole']}"] = 1
        input_data[f"MaritalStatus_{categorical_features['MaritalStatus']}"] = 1

        df = pd.DataFrame([input_data])
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)

        #  print statement to check the input data
        print(f"Input Data: {data}")

        pred = service.predict(data)
        st.success(
            "Predicted Employee Attrition Probability (0 - 1): {:.2f}".format(
                pred[0]
            )
        )

        print("Prediction completed")

        # Stop the service after prediction
        if service_started:
            stop_service(service)

if __name__ == "__main__":
    run_deployment()
