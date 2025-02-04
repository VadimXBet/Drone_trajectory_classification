import os
import re
import cv2
import math
import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from feature_calculation import *

def append_data_for_dataframe(x_centers, y_centers, batch_size, label, result_df):
    x_batch = batch(x_centers, batch_size)
    y_batch = batch(y_centers, batch_size)
    for x_center_batch, y_center_batch in zip(x_batch, y_batch):
        TA = turn_angle(x_center_batch, y_center_batch)
        C = curvature(x_center_batch, y_center_batch)
        V = velocity(x_center_batch, y_center_batch)
        A = acceleration(x_center_batch, y_center_batch)
        CD = CDF(x_center_batch, y_center_batch)
        buffer_df = pd.DataFrame([{'turn_angle' : TA, 
                        'curvature' : C, 
                        'velocity' : V, 
                        'acceleration' : A, 
                        'CDF' : CD,
                        'label' : label}])
        result_df._append(buffer_df, ignore_index=True)
        result_df = result_df._append(buffer_df, ignore_index=True)
    return result_df

def data_writer(batch_size = 15):
    colnames = ['turn_angle', 'curvature', 'velocity', 
                'acceleration', 'CDF', 'label']
    result_df = pd.DataFrame(columns = colnames)

    for label in ['drones', 'birds']:
        data_path = os.path.join('data', label, 'gt')

        for file in os.listdir(os.path.join('data', label, 'videos')):
            video_path = os.path.join('data', label, 'videos', file)
            cap = cv2.VideoCapture(video_path)
            frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            file_name = file[:-4]
            # print(f'File {file_name} is processing')

            if os.path.exists(os.path.join(data_path, file_name+'_LABELS.csv')):
                df = pd.read_csv(os.path.join(data_path, file_name+'_LABELS.csv'))
                df = df.drop(columns=['Unnamed: 0', 'time'], axis=1)
                size = len(df['object_1'])

                for name_column in df.columns:
                    x_centers, y_centers = [], []
                    df[name_column] = df[name_column].apply(lambda x: x.strip("[]").split(", "))
                    for i in range(size):
                        if not math.isnan(float(df[name_column][i][0])):
                            xtl, ytl, h, w = float(df[name_column][i][0]), float(df[name_column][i][1]), float(df[name_column][i][3]), float(df[name_column][i][2])
                            xc, yc = xtl+w/2, ytl+h/2
                            x_centers.append(xc/frame_width)
                            y_centers.append(yc/frame_height)
                    result_df = append_data_for_dataframe(x_centers, y_centers, batch_size, label, result_df)
            
            elif os.path.exists(os.path.join(data_path, file_name+'_LABELS.txt')):
                df = np.loadtxt(os.path.join(data_path, file_name+'_LABELS.txt'), dtype="float", delimiter=",", usecols =(0,1,2,3,4,5))
                for id in set(df[:, 1]):
                    x_centers, y_centers = [], []
                    id_coord = df[df[:, 1] == id] 
                    for i in range(len(id_coord)):
                        xtl, ytl, w, h = id_coord[i][2], id_coord[i][3], id_coord[i][4], id_coord[i][5]
                        xc, yc = xtl+w/2, ytl+h/2
                        x_centers.append(xc/frame_width)
                        y_centers.append(yc/frame_height)
                    result_df = append_data_for_dataframe(x_centers, y_centers, batch_size, label, result_df)
            # else:    
            #     print(' has not scv or txt file\n')
            
        result_df.to_csv('data.csv', index=False)
    return result_df

def test_data_writer(batch_size = 15):
    colnames = ['turn_angle', 'curvature', 'velocity', 
                'acceleration', 'CDF', 'label']
    result_df = pd.DataFrame(columns = colnames)

    path = 'test_data\gt'
    for file in os.listdir(path):
        frame_width = 1920
        frame_height = 1072
        # print(f'File {file} is processing')
        
        df = np.loadtxt(os.path.join(path, file), delimiter=",", usecols =(0,1,2,3,4,5,6))
        for id in set(df[:, 1]):
            x_centers, y_centers = [], []
            id_coord = df[df[:, 1] == id] 
            label = 'drones' if id_coord[0][6] == 1 else 'birds'
            for i in range(len(id_coord)):
                xtl, ytl, w, h = id_coord[i][2], id_coord[i][3], id_coord[i][4], id_coord[i][5]
                xc, yc = xtl+w/2, ytl+h/2
                x_centers.append(xc/frame_width)
                y_centers.append(yc/frame_height)
            result_df = append_data_for_dataframe(x_centers, y_centers, batch_size, label, result_df)
            
        result_df.to_csv('test_data.csv', index=False)
    return result_df

def make_data(df):
    df['label'] = df['label'].apply(lambda x: int(0) if x == 'drones' else int(1))
    return train_test_split(df.iloc[:, 0:5], df.iloc[:, 5], train_size=0.7)

def print_score(clf, X_train, y_train, X_test, y_test):
    pred_train = clf.predict(X_train)
    print(f"Train Result:\n================================================")
    print(f"Train Accuracy Score: {accuracy_score(y_train, pred_train)}%")
    print("_______________________________________________")
    print(f"Train Precision Score (drones): {precision_score(y_train, pred_train, pos_label=0)}%")
    print("_______________________________________________")
    print(f"Train Precision Score (birds): {precision_score(y_train, pred_train, pos_label=1)}%")
    print("_______________________________________________")

    pred_test = clf.predict(X_test)
    print(f"Test Result:\n================================================")
    print(f"Test Accuracy Score: {accuracy_score(y_test, pred_test)}%")
    print("_______________________________________________")
    print(f"Test Precision Score (drones): {precision_score(y_test, pred_test, pos_label=0)}%")
    print("_______________________________________________")
    print(f"Test Precision Score (birds): {precision_score(y_test, pred_test, pos_label=1)}%")
    print("_______________________________________________")

def OneBigClassifier(X_train, y_train, X_test, y_test, BATCH_SIZE, result_df):
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [1, 0.1, 0.01],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, cv=4, n_jobs=-1)
    grid.fit(X_train, y_train)
    SVM_pred_test = grid.predict(X_test)
    joblib.dump(grid, f"weights/SVM_model_{BATCH_SIZE}.pkl") 

    param_grid = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'n_estimators': [50, 100],
        'max_depth': [5, 10, 20]
    }
    grid = GridSearchCV(RF(random_state=42), param_grid, refit=True, verbose=3, cv=4, n_jobs=-1)
    grid.fit(X_train.values, y_train)
    RF_pred_test = grid.predict(X_test)
    joblib.dump(grid, f"weights/RF_model_{BATCH_SIZE}.pkl") 

    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 0.2],
    }
    grid = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid, refit=True, verbose=3, cv=4, n_jobs=-1)
    grid.fit(X_train.values, y_train)
    AB_pred_test = grid.predict(X_test)
    joblib.dump(grid, f"weights/AB_model_{BATCH_SIZE}.pkl") 

    model = make_pipeline(StandardScaler(), XGBClassifier())
    parameters = {
        'max_depth': [5, 10, 20],
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    grid = GridSearchCV(XGBClassifier(), param_grid, refit=True, verbose=3, cv=4, n_jobs=-1)
    grid.fit(X_train.values, y_train)
    GB_pred_test = grid.predict(X_test)
    joblib.dump(grid, f"weights/GB_model_{BATCH_SIZE}.pkl") 

    param_grid = {
        'loss': ['log_loss', 'exponential'],
        'learning_rate': [0.01, 0.1, 0.2],
        'criterion': ['friedman_mse', 'squared_error'],
        'n_estimators': [50, 100],
        'max_depth': [5, 10, 20]
    }
    grid = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, refit=True, verbose=3, cv=4, n_jobs=-1)
    grid.fit(X_train.values, y_train)
    XGB_pred_test = grid.predict(X_test)
    joblib.dump(grid, f"weights/XGB_model_{BATCH_SIZE}.pkl") 

    param_grid = {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [4, 6, 8]
    }
    grid = GridSearchCV(CatBoostClassifier(), param_grid, refit=True, verbose=3, cv=4, n_jobs=-1)
    grid.fit(X_train.values, y_train)
    CB_pred_test = grid.predict(X_test)
    joblib.dump(grid, f"weights/CB_model_{BATCH_SIZE}.pkl") 

    buffer_df = pd.DataFrame([{'batchs size' : BATCH_SIZE, 
                                'SVM Test Accuracy' : accuracy_score(y_test, SVM_pred_test),
                                'SVM Test Precision (drones)' : precision_score(y_test, SVM_pred_test, pos_label=0),
                                'SVM Test Precision (birds)' : precision_score(y_test, SVM_pred_test, pos_label=1),
                                'RANDOM_FOREST Test Accuracy' : accuracy_score(y_test, RF_pred_test),
                                'RANDOM_FOREST Test Precision (drones)' : precision_score(y_test, RF_pred_test, pos_label=0),
                                'RANDOM_FOREST Test Precision (birds)' : precision_score(y_test, RF_pred_test, pos_label=1),
                                'ADABoost Test Accuracy' : accuracy_score(y_test, AB_pred_test),
                                'ADABoost Test Precision (drones)' : precision_score(y_test, AB_pred_test, pos_label=0),
                                'ADABoost Test Precision (birds)' : precision_score(y_test, AB_pred_test, pos_label=1),
                                'GradientBoosting Test Accuracy' : accuracy_score(y_test, GB_pred_test),
                                'GradientBoosting Test Precision (drones)' : precision_score(y_test, GB_pred_test, pos_label=0),
                                'GradientBoosting Test Precision (birds)' : precision_score(y_test, GB_pred_test, pos_label=1),
                                'XGBoost Test Accuracy' : accuracy_score(y_test, XGB_pred_test),
                                'XGBoost Test Precision (drones)' : precision_score(y_test, XGB_pred_test, pos_label=0),
                                'XGBoost Test Precision (birds)' : precision_score(y_test, XGB_pred_test, pos_label=1),
                                'CatBoost Test Accuracy' : accuracy_score(y_test, CB_pred_test),
                                'CatBoost Test Precision (drones)' : precision_score(y_test, CB_pred_test, pos_label=0),
                                'CatBoost Test Precision (birds)' : precision_score(y_test, CB_pred_test, pos_label=1)}])
    result_df._append(buffer_df, ignore_index=True)
    result_df = result_df._append(buffer_df, ignore_index=True)
    return result_df

if __name__ == '__main__':
    colnames = ['batchs size', 
                'SVM Test Accuracy', 'SVM Test Precision (drones)', 'SVM Test Precision (birds)',
                'RANDOM_FOREST Test Accuracy', 'RANDOM_FOREST Test Precision (drones)', 'RANDOM_FOREST Test Precision (birds)',
                'ADABoost Test Accuracy', 'ADABoost Test Precision (drones)', 'ADABoost Test Precision (birds)',
                'GradientBoosting Test Accuracy', 'GradientBoosting Test Precision (drones)', 'GradientBoosting Test Precision (birds)',
                'XGBoost Test Accuracy', 'XGBoost Test Precision (drones)', 'XGBoost Test Precision (birds)',
                'CatBoost Test Accuracy', 'CatBoost Test Precision (drones)', 'CatBoost Test Precision (birds)']
    result_metrics_df = pd.DataFrame(columns = colnames)

    for BATCH_SIZE in range(5, 26, 5):
        print(BATCH_SIZE)
        train_data = data_writer(BATCH_SIZE)
        test_data = test_data_writer(BATCH_SIZE)
        data = pd.concat([train_data, test_data], ignore_index=True)
        X_train, X_test, y_train, y_test = make_data(data)
        result_metrics_df = OneBigClassifier(X_train, y_train, X_test, y_test, BATCH_SIZE, result_metrics_df)

    result_metrics_df.to_csv('result metrics.csv', index=False)

    read_file = pd.read_csv('result metrics.csv')
    read_file.to_excel('result metrics.xlsx', index=None, header=True)
