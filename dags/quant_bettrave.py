from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 4, 20),
    'retries': 1,
}

dag = DAG(
    'quantity_bettrave_pipeline',
    default_args=default_args,
    description='A DAG to improve data quality',
    schedule_interval='@weekly',
    catchup=False
)

def extract_data():
    file_path = '/opt/airflow/dags/caract_so.csv'
    df = pd.read_csv(file_path)
    return df.to_dict(orient='records')

def transform_data(**kwargs):
    ti = kwargs['ti']
    df_list = ti.xcom_pull(task_ids='extract_data')
    df = pd.DataFrame(df_list)
    numeric_columns = ['N', 'P', 'K', 'quantite']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    return df.to_dict(orient='records')

def clean_data(**kwargs):
    ti = kwargs['ti']
    df_list = ti.xcom_pull(task_ids='transform_data')
    df = pd.DataFrame(df_list)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df.to_dict(orient='records')

def validate_data(**kwargs):
    ti = kwargs['ti']
    df_list = ti.xcom_pull(task_ids='clean_data')  # Récupérer les données de la tâche clean_data

    if not isinstance(df_list, list):
        raise ValueError("Expected a list from clean_data task.")

    # Convertir la liste de dictionnaires en DataFrame
    df = pd.DataFrame(df_list)

    validation_results = {}
    
    validation_rules = {
        'temperature': {'dtype': 'float', 'min_value': 0, 'max_value': 50},
        'humidity': {'dtype': 'float', 'min_value': 0, 'max_value': 100},
        'ph': {'dtype': 'float', 'min_value': 0, 'max_value': 14},
        'rainfall': {'dtype': 'float', 'min_value': 0},
        'quantite': {'dtype': 'int', 'min_value': 0}
    }
    
    for col_name, rules in validation_rules.items():
        col_dtype = df[col_name].dtype
        
        if rules['dtype'] == 'int' and col_dtype != 'int64':
            validation_results[col_name + '_dtype_valid'] = False
        elif rules['dtype'] == 'float' and col_dtype != 'float64':
            validation_results[col_name + '_dtype_valid'] = False
        else:
            validation_results[col_name + '_dtype_valid'] = True
        
        if 'min_value' in rules and 'max_value' in rules:
            valid_range = (df[col_name] >= rules['min_value']) & (df[col_name] <= rules['max_value'])
            validation_results[col_name + '_range_valid'] = valid_range.all()
        elif 'min_value' in rules:
            valid_min = (df[col_name] >= rules['min_value'])
            validation_results[col_name + '_range_valid'] = valid_min.all()
    
    print("Validation Results:")
    for key, value in validation_results.items():
        print(f"{key}: {value}")
    
    if not all(validation_results.values()):
        raise ValueError("Data validation failed.")
    print(df.columns)
    return df.to_dict(orient='records')
    
def load_data(**kwargs):
    ti = kwargs['ti']
    df_list = ti.xcom_pull(task_ids='validate_data')
    df = pd.DataFrame(df_list)
    # Sauvegarder les données nettoyées et validées dans une base de données
    from sqlalchemy import create_engine
    engine = create_engine('postgresql://postgres:postgres@localhost/postgres')
    df.to_sql('cleaned_data', engine, if_exists='replace', index=False)

def train_task(**kwargs):
    ti = kwargs['ti']
    df_list = ti.xcom_pull(task_ids='validate_data')
    df = pd.DataFrame(df_list)
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    
    X = df.drop(columns=['quantite'])
    y = df['quantite']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    import pickle
    pickle.dump(model,open('/opt/airflow/trained_models/trained_model.pkl','wb'))
    X_test.to_csv('/opt/airflow/trained_models/X_test.csv', index=False)
    ti.xcom_push(key='X_test_filename', value='/opt/airflow/trained_models/X_test.csv')

def deploy_app():
    import subprocess
    # Commande pour lancer l'application Flask comme un processus subprocess
    subprocess.Popen(['python', 'C:/Users/PcPack/hha/dags/app.py'])


extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    provide_context=True,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    provide_context=True,
    dag=dag,
)

clean_task = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data,
    provide_context=True,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    provide_context=True,
    dag=dag,
)
load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    provide_context=True,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_task',
    python_callable=train_task,
    provide_context=True,
    dag=dag,
)

deploy_task = PythonOperator(
    task_id='deploy_app',
    python_callable=deploy_app,
    provide_context=True,
    dag=dag,
)

# Define task dependencies
extract_task >> transform_task >> clean_task >> validate_task >>load_task >> train_task >> deploy_task
