from zenml.client import Client

from pipeline.tarining_pipeline import train_pipeline

if __name__ =="__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path=r"/home/Employee-Attrition-Rate-MLOps/data/HR-Employee-Attrition.csv")
    

# mlflow ui --backend-store-uri "file:/root/.config/zenml/local_stores/1fb2f610-b8a4-4a4a-ba9b-0ec747164644/mlruns"