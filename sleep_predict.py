# basic
import streamlit as st
import pandas as pd
import numpy as np
import os
import copy

# classifiers
from sklearn.ensemble import RandomForestClassifier

# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# mertics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# other
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import dates as plotdates
from matplotlib.figure import Figure

# definitions
HERE_PATH = os.path.dirname(__file__)  # "D:/Progamming/Python/Web_api/Streamlit/Test

RAW_DATA_PATH = os.path.join(HERE_PATH, "datatest/raw_data.csv")
PROCESSED_DATA_PATH = os.path.join(HERE_PATH, "datatest/processed_data.csv")
FILTERED_DATA_PATH = os.path.join(HERE_PATH, "datatest/processed_data_selected_features.csv")

MULTICLASS_CLASSIFIER_PATH = os.path.join(HERE_PATH, "classifiers/stage_sleep.sav")
BINARY_CLASSIFIER_PATH = os.path.join(HERE_PATH, "classifiers/binary_sleep.sav")

TEST_DAY_PATH = os.path.join(HERE_PATH, "datatest/test_days/")

day_choices = ["Day 1", "Day 2", "Day 3", "Day 4"]
day_options = [ "2021_01_15.csv", "2021_01_16.csv", "2021_01_19.csv", "2021_01_21.csv" ]
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Sleep prediction App',
    layout='wide')
#---------------------------------#
# functions
def load_file(path, index=False):
    if index:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index)
    else:
        df = pd.read_csv(path)
    if path == RAW_DATA_PATH:
        df = df.drop(["time"], axis=1)
    return df

def get_classifier(flag):
    if flag == "sleep/wake":
        conf_plot_labels = ["awake", "sleep"]
        clf = RandomForestClassifier(criterion='entropy', n_estimators=100)
    if flag == "wake/light/deep/REM":
        clf = RandomForestClassifier(criterion='entropy', n_estimators=800)
        conf_plot_labels = ["deep", "light", "rem", "wake"]
    return clf, conf_plot_labels

def load_classifier(flag):
    if flag == "sleep/wake":
        conf_plot_labels = ["awake", "sleep"]
        clf = pickle.load(open(BINARY_CLASSIFIER_PATH, 'rb'))
    if flag == "wake/light/deep/REM":
        conf_plot_labels = ["deep", "light", "rem", "wake"]
        clf = pickle.load(open(MULTICLASS_CLASSIFIER_PATH, 'rb'))
    return clf, conf_plot_labels

@st.cache(allow_output_mutation=True)
def fit_model(clf, X, y):
    clf.fit(X, y)
    return clf

def detection_type(flag, df):
    if flag == "sleep/wake":
        df["label"] = df["label"].replace(["wake", "light", "deep", "rem"], "sleep")
        df["label"] = df["label"].replace(["sleep"], 0)
        df["label"] = df["label"].replace(["awake"], 1)
    if flag == "wake/light/deep/REM":
        df["label"] = df["label"].replace(["awake"], "wake")
    return df

def get_plot_dates(data):
    date = []
    for row in data:
        date.append(plotdates.date2num(row))
    return date

def change_labels(flag, y_pred, y_true):
    if flag == "sleep/wake":
        y_indexes = ["awake", "sleep"]
        y_pred = pd.Series(y_pred)
        y_pred = y_pred.replace(1, "awake")
        y_pred = y_pred.replace(0, "sleep")
        y_true = y_true.replace(1, "awake")
        y_true = y_true.replace(0, "sleep")
    if flag == "wake/light/deep/REM":
        y_indexes = ["wake", "light", "deep", "rem"]
        #0 - wake, 1 - light, 2 - deep, 3 - rem, 4 - awake
        y_pred = pd.Series(y_pred)
        # y_pred = y_pred.replace("wake", 0)
        # y_pred = y_pred.replace("light", 1)
        # y_pred = y_pred.replace("deep", 2)
        # y_pred = y_pred.replace("rem", 3)

        # y_true = y_true.replace("wake", 0)
        # y_true = y_true.replace("light", 1)
        # y_true = y_true.replace("deep", 2)
        # y_true = y_true.replace("rem", 3)
    return y_indexes, y_pred, y_true

def make_conf_matrix(clf, X, y, labels):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    disp = plot_confusion_matrix(clf, X, y,
                                 display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize=None,
                                 ax=ax)
    disp.ax_.set_ylabel("Real labels")
    disp.ax_.set_xlabel("Predicted labels")
    return fig

def make_comprasion_plot(y_pred, y_true, class_option, date_values):
    y_indexes, y_pred, y_true = change_labels(class_option, y_pred, y_true)
    plot_dates = get_plot_dates(date_values)

    fig = Figure()
    ax = fig.add_subplot(2,1,1)
    ax.xaxis.set_major_locator(plotdates.HourLocator())
    ax.xaxis.set_major_formatter(plotdates.DateFormatter('%H'))
    # ax.set_xlabel("Time[h]")
    ax.set_ylabel("Fitbit")
    ax.plot(plot_dates, y_true)

    ax = fig.add_subplot(2,1,2)
    ax.xaxis.set_major_locator(plotdates.HourLocator())
    ax.xaxis.set_major_formatter(plotdates.DateFormatter('%H'))
    ax.set_xlabel("Time[h]")
    ax.set_ylabel("Prediction")
    ax.plot(plot_dates, y_pred)

    fig.suptitle('Comprasion of true and predicted labels', fontsize=16)
    return fig

def find_start(data):
    for i in range(len(data)-19):
        if data[i] == data[i+1] == data[i+2] == data[i+3] == data[i+4]== data[i+5]== data[i+6]== data[i+7]== data[i+8]== data[i+9]==data[i+10] == data[i+11] == data[i+12] == data[i+13] == data[i+14]== data[i+15]== data[i+16]== data[i+17]== data[i+18]== data[i+19] and data[i]==0:
            start = i
            return start

def find_end(data):
    for i in range(len(data)-1, 4, -1):
        if data[i] == data[i-1] == data[i-2] == data[i-3] == data[i-4] == data[i-5] == data[i-6] == data[i-7] == data[i-8] == data[i-9]==data[i-10] == data[i-11] == data[i-12] == data[i-13] == data[i-14]== data[i-15]== data[i-16]== data[i-17]== data[i-18]== data[i-19] and data[i]==0:
            end = i
            return end

#---------------------------------#
# main function
def build():
    df = load_file(FILTERED_DATA_PATH)
    df = detection_type(class_option, df)

    clf, conf_plot_labels = get_classifier(class_option)
    y = df["label"]
    X = df.drop(["label"], axis=1)
    clf = fit_model(clf, X, y)

    # clf, conf_plot_labels = load_classifier(class_option)

    df_day = load_file(TEST_DAY_PATH + day_options[day_option], index=True)
    df_day = detection_type(class_option, df_day)

    #---------------------------------#
    st.write("**1.0** Data")
    if st.checkbox('Show data'):
        st.write(df_day)

    #---------------------------------#
    st.subheader('**1.1** Prediction accuracy')
    X_true = df_day.drop(["label"], axis=1)
    y_true = df_day["label"]

    y_pred = clf.predict(X_true)
    acc = accuracy_score(y_true, y_pred)
    st.write(acc)

    #---------------------------------#
    st.subheader('**1.2** Confusion matrix')
    fig = make_conf_matrix(copy.deepcopy(clf), X_true, y_true, conf_plot_labels)
    st.write(fig)

    #---------------------------------#
    st.subheader('**1.3** Real and predicted comprasion')
    fig = make_comprasion_plot(y_pred, y_true, class_option, df_day.index.values)
    st.pyplot(fig)

    #---------------------------------#
    if class_option == "sleep/wake":
        st.subheader('**1.4** Estimated sleep time')
        true_list, pred_list = [], []

        fall_asleep_time = y_true.index[int(find_start(y_true))].strftime('%H:%M:%S')
        true_list.append(fall_asleep_time)
        fall_asleep_time = y_true.index[int(find_start(y_pred))].strftime('%H:%M:%S')
        pred_list.append(fall_asleep_time)

        wakeup_time = y_true.index[int(find_end(y_true))].strftime('%H:%M:%S')
        true_list.append(wakeup_time)
        wakeup_time = y_true.index[int(find_end(y_pred))].strftime('%H:%M:%S')
        pred_list.append(wakeup_time)

        tst_real = int(find_end(y_true)-find_start(y_true)) * 30 / 60
        true_list.append(tst_real)
        tst = int(find_end(y_pred)-find_start(y_pred)) * 30 / 60
        pred_list.append(tst)

        df_time = pd.DataFrame({"True" : true_list, "Prediction": pred_list}, index=["fall asleep time", "wake up time", "total sleep time (TST)"])
        st.write(df_time)

# ----------------------#
# Sidebar
# Sidebar select mode
# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write("Lol")

# Sidebar - Specify parameter settings
with st.sidebar.subheader("Select days"):
    day_option = st.selectbox(
        'What day would you like to choose?',
        list(range(len(day_choices))), format_func=lambda x: day_choices[x])

with st.sidebar.subheader("Select classes"):
    class_option = st.selectbox(
        'What classes would you like to choose?',
        ["sleep/wake", "wake/light/deep/REM"])

# with st.sidebar.header('1. Set test split Parameters'):
#     param_split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
#     param_random_state = st.sidebar.slider('Random state of train split', 0, 100, 42, 1)






# with st.sidebar.subheader('2.1. Learning Parameters'):
#     parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
#     parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
#     parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
#     parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

# with st.sidebar.subheader('2.2. General Parameters'):
#     parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
#     parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
#     parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
#     parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
#     parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])



# ----------------------#
# Main page 
st.title('Sleep stage prediction based on Fitbit sleep predictions')
build()