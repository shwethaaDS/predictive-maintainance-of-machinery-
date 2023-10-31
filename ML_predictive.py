import tkinter as tk
from tkinter import Text
from tkinter import messagebox,ttk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils import compute_sample_weight
from sklearn.svm import SVC
from xgboost import XGBClassifier

#path = 'F:/LastMile/'
df = pd.read_csv("predictive_maintainence.csv") 

target_name = 'Failure Type'

#path = 'F:/LastMile/'

root = tk.Tk()
root.title("Predictive Maintenance Dashboard")
root.geometry("800x600")  
root.configure(bg='light blue')
style = ttk.Style()
style.configure("TButton", padding=10, font=("Helvetica", 12))
style.configure("TLabel", font=("Helvetica", 14))

main_frame = ttk.Frame(root)
main_frame.pack()

def show_result_window(title, results):
    result_label = ttk.Label(main_frame, text=results)
    result_label.pack()
    root.update_idletasks()
    
def run_logistic_regression():
    def logistic_regression_code():
        
        def print_missing_values(df):
            null_df = pd.DataFrame(df.isna().sum(), columns=['null_values']).sort_values(['null_values'], ascending=False)
            fig = plt.subplots(figsize=(16, 6))
            ax = sns.barplot(data=null_df, x='null_values', y=null_df.index, color='royalblue')
            pct_values = [' {:g}'.format(elm) + ' ({:.1%})'.format(elm/len(df)) for elm in list(null_df['null_values'])]
            ax.set_title('Overview of missing values')
            ax.bar_label(container=ax.containers[0], labels=pct_values, size=12)

        if df.isna().sum().sum() > 0:
            print_missing_values(df)
        else:
            print('No missing values')

        for col_name in df.columns:
            if df[col_name].isna().sum() / df.shape[0] > 0.05:
                df.drop(columns=[col_name], inplace=True) 
        df_base = df.drop(columns=['Product ID', 'UDI'])

        df_base.rename(columns={'Air temperature [K]': 'air_temperature', 
                                'Process temperature [K]': 'process_temperature', 
                                'Rotational speed [rpm]':'rotational_speed', 
                                'Torque [Nm]': 'torque', 
                                'Tool wear [min]': 'tool_wear'}, inplace=True)

        px.histogram(df_base, y="Failure Type", color="Failure Type") 

        sns.pairplot(df_base, height=1, hue='Failure Type')

        numeric_columns = df_base.select_dtypes(include=[np.number]).columns
        correlation_matrix = df_base[numeric_columns].corr()

        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_matrix, cbar=True, fmt='.1f', vmax=0.8, annot=True, cmap='Blues')
        plt.show()

        def data_preparation(df_base, target_name):
            df = df_base.dropna()

            df['target_name_encoded'] = df[target_name].replace({'No Failure': 0, 'Power Failure': 1, 'Tool Wear Failure': 2, 'Overstrain Failure': 3, 'Random Failures': 4, 'Heat Dissipation Failure': 5})
            df['Type'].replace({'L': 0, 'M': 1, 'H': 2}, inplace=True)
            X = df.drop(columns=[target_name, 'target_name_encoded'])
            y = df['target_name_encoded']  

            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

            print('Train: ', X_train.shape, y_train.shape)
            print('Test: ', X_test.shape, y_test.shape)
            return X, y, X_train, X_test, y_train, y_test

        X, y, X_train, X_test, y_train, y_test = data_preparation(df_base, target_name)
        weight_train = compute_sample_weight('balanced', y_train)
        weight_test = compute_sample_weight('balanced', y_test)

        logistic_reg = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')

        logistic_reg.fit(X_train, y_train, sample_weight=weight_train)

        score = logistic_reg.score(X_test, y_test, sample_weight=weight_test)

        y_pred = logistic_reg.predict(X_test)

        # Print a multi-class confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cnf_matrix, columns=np.unique(y_test), index=np.unique(y_test))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure(figsize=(5, 3))
        sns.set(font_scale=1.1)  # for label size
        sns.heatmap(df_cm, cbar=True, cmap="inferno", annot=True, fmt='.0f')


        results_log = classification_report(y_test, y_pred)
        
        return results_log

    results = logistic_regression_code()
    result_text.delete(1.0, "end")
    result_text.insert("end", results)
    result_text.insert("end", "Logistic Model Generated!")

def run_svm():
    def svm_code():
        
        def print_missing_values(df):
            null_df = pd.DataFrame(df.isna().sum(), columns=['null_values']).sort_values(['null_values'], ascending=False)
            fig = plt.subplots(figsize=(16, 6))
            ax = sns.barplot(data=null_df, x='null_values', y=null_df.index, color='royalblue')
            pct_values = [' {:g}'.format(elm) + ' ({:.1%})'.format(elm/len(df)) for elm in list(null_df['null_values'])]
            ax.set_title('Overview of missing values')
            ax.bar_label(container=ax.containers[0], labels=pct_values, size=12)

        if df.isna().sum().sum() > 0:
            print_missing_values(df)
        else:
            print('No missing values')

        # Drop all columns with more than 5% missing values
        for col_name in df.columns:
            if df[col_name].isna().sum() / df.shape[0] > 0.05:
                df.drop(columns=[col_name], inplace=True) 

        # Drop ID columns
        df_base = df.drop(columns=['Product ID', 'UDI'])

        # Adjust column names
        df_base.rename(columns={'Air temperature [K]': 'air_temperature', 
                                'Process temperature [K]': 'process_temperature', 
                                'Rotational speed [rpm]':'rotational_speed', 
                                'Torque [Nm]': 'torque', 
                                'Tool wear [min]': 'tool_wear'}, inplace=True)

        
        sns.pairplot(df_base, height=1, hue='Failure Type')
        numeric_columns = df_base.select_dtypes(include=[np.number]).columns
        correlation_matrix = df_base[numeric_columns].corr()

        plt.figure(figsize=(4, 4))
        sns.heatmap(correlation_matrix, cbar=True, fmt='.1f', vmax=0.8, annot=True, cmap='Blues')
        plt.show()

        
        def data_preparation(df_base, target_name):
            df = df_base.dropna()

            df['target_name_encoded'] = df[target_name].replace({'No Failure': 0, 'Power Failure': 1, 'Tool Wear Failure': 2, 'Overstrain Failure': 3, 'Random Failures': 4, 'Heat Dissipation Failure': 5})
            df['Type'].replace({'L': 0, 'M': 1, 'H': 2}, inplace=True)
            X = df.drop(columns=[target_name, 'target_name_encoded'])
            y = df['target_name_encoded']  

            
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

           
            print('Train: ', X_train.shape, y_train.shape)
            print('Test: ', X_test.shape, y_test.shape)
            return X, y, X_train, X_test, y_train, y_test

        
        X, y, X_train, X_test, y_train, y_test = data_preparation(df_base, target_name)

        
        weight_train = compute_sample_weight('balanced', y_train)
        weight_test = compute_sample_weight('balanced', y_test)

        
        svm_clf = SVC(kernel='linear', decision_function_shape='ovr', class_weight='balanced')

        
        svm_clf.fit(X_train, y_train, sample_weight=weight_train)

        
        score = svm_clf.score(X_test, y_test, sample_weight=weight_test)

        
        y_pred = svm_clf.predict(X_test)
        
        cnf_matrix = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cnf_matrix, columns=np.unique(y_test), index=np.unique(y_test))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure(figsize=(5, 3))
        sns.set(font_scale=1.1)  # for label size
        sns.heatmap(df_cm, cbar=True, cmap="inferno", annot=True, fmt='.0f')

        results_svm = classification_report(y_test, y_pred)
        return results_svm  
    
    results = svm_code()
    result_text.delete(1.0, "end") 
    result_text.insert("end", results)  
    result_text.insert("end", "\nSVM Model Generated!")

def run_xgboost():
    def xgboost_code():
        
        def print_missing_values(df):
            null_df = pd.DataFrame(df.isna().sum(), columns=['null_values']).sort_values(['null_values'], ascending=False)
            fig = plt.subplots(figsize=(16, 6))
            ax = sns.barplot(data=null_df, x='null_values', y=null_df.index, color='royalblue')
            pct_values = [' {:g}'.format(elm) + ' ({:.1%})'.format(elm/len(df)) for elm in list(null_df['null_values'])]
            ax.set_title('Overview of missing values')
            ax.bar_label(container=ax.containers[0], labels=pct_values, size=12)

        if df.isna().sum().sum() > 0:
            print_missing_values(df)
        else:
            print('No missing values')

        
        for col_name in df.columns:
            if df[col_name].isna().sum() / df.shape[0] > 0.05:
                df.drop(columns=[col_name], inplace=True)

        
        df_base = df.drop(columns=['Product ID', 'UDI'])

       
        df_base.rename(columns={'Air temperature [K]': 'air_temperature', 
                                'Process temperature [K]': 'process_temperature', 
                                'Rotational speed [rpm]':'rotational_speed', 
                                'Torque [Nm]': 'torque', 
                                'Tool wear [min]': 'tool_wear'}, inplace=True)

        
        sns.pairplot(df_base, height=1, hue='Failure Type')
        numeric_columns = df_base.select_dtypes(include=[np.number]).columns
        correlation_matrix = df_base[numeric_columns].corr()

        plt.figure(figsize=(4, 4))
        sns.heatmap(correlation_matrix, cbar=True, fmt='.1f', vmax=0.8, annot=True, cmap='Blues')
        plt.show()

        
        def data_preparation(df_base, target_name):
            df = df_base.dropna()

            df['target_name_encoded'] = df[target_name].replace({'No Failure': 0, 'Power Failure': 1, 'Tool Wear Failure': 2, 'Overstrain Failure': 3, 'Random Failures': 4, 'Heat Dissipation Failure': 5})
            df['Type'].replace({'L': 0, 'M': 1, 'H': 2}, inplace=True)
            X = df.drop(columns=[target_name, 'target_name_encoded'])
            y = df['target_name_encoded']  

            
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

            
            print('Train: ', X_train.shape, y_train.shape)
            print('Test: ', X_test.shape, y_test.shape)
            return X, y, X_train, X_test, y_train, y_test

        
        X, y, X_train, X_test, y_train, y_test = data_preparation(df_base, target_name)

        
        weight_train = compute_sample_weight('balanced', y_train)
        weight_test = compute_sample_weight('balanced', y_test)

        xgb_clf = XGBClassifier(booster='gbtree', 
                               tree_method='hist',  
                               sampling_method='uniform',  
                               eval_metric='aucpr', 
                               objective='multi:softmax', 
                               num_class=6)

       
        xgb_clf.fit(X_train, y_train.ravel(), sample_weight=weight_train)

        
        score = xgb_clf.score(X_test, y_test.ravel(), sample_weight=weight_test)

        
        y_pred = xgb_clf.predict(X_test)
                
        cnf_matrix = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cnf_matrix, columns=np.unique(y_test), index=np.unique(y_test))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure(figsize=(5, 3))
        sns.set(font_scale=1.1) 
        sns.heatmap(df_cm, cbar=True, cmap="inferno", annot=True, fmt='.0f')

        results_log = classification_report(y_test, y_pred)

       
        results_log = classification_report(y_test, y_pred)
        return results_log  

    results = xgboost_code()
    result_text.delete(1.0, "end")  
    result_text.insert("end", results)  
    result_text.insert("end", "\nXGBoost Model Generated!")



svm_button = tk.Button(root, text="Generate SVM Model", command=run_svm,bg='lightyellow')
svm_button.pack(pady=10)

logistic_regression_button = tk.Button(root, text="Generate Logistic Regression Model", command=run_logistic_regression, bg='lightyellow')
logistic_regression_button.pack(pady=10)



xgboost_button = tk.Button(root, text="Generate XGBoost Model", command=run_xgboost,bg='lightyellow')
xgboost_button.pack(pady=10)
def exit_application():
    root.destroy()


exit_button = tk.Button(root, text="Exit", command=exit_application, bg='lightyellow')
exit_button.pack(pady=10)





result_text = Text(root, height=20, width=80)
result_text.pack(pady=10)


root.mainloop()