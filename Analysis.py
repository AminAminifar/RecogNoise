import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier

# def get_rr_intervals(r_peak_vector):
#     rr_intervals_vec = r_peak_vector[1:]
#     rr_intervals_vec -= r_peak_vector[:-1]
#     return rr_intervals_vec

def data_preparation(data_df):
    patients = data_df['patient'].unique()
    patients_train, patients_test = train_test_split(patients, test_size=0.3)  # , random_state = 42

    algorithms = data_df['algorithm'].unique()
    channels = data_df['channel'].unique()

    # each record is splitted to 30*3 slices (each slice is 20-sec (1 minute/3) data)
    num_windows = data_df['window_num'].max()

    # determine the length of vector for each alg.
    indices_of_patients_train_in_df = data_df[data_df['patient'].isin(patients_train)].index
    mean_value = data_df.loc[indices_of_patients_train_in_df, 'rr_intervals_vec_length'].mean()
    std_value = data_df.loc[indices_of_patients_train_in_df, 'rr_intervals_vec_length'].std()
    vec_length = int(mean_value) + int(std_value)

    # make empty (zero filled) matrices/vectors for data
    num_rows_train = len(channels) * num_windows * len(patients_train)
    num_rows_test = len(channels) * num_windows * len(patients_test)
    num_columns = vec_length * len(algorithms)

    X_train, X_test, y_train, y_test = (np.zeros((num_rows_train, num_columns)), np.zeros((num_rows_test, num_columns)),
                                        np.zeros(num_rows_train), np.zeros(num_rows_test))

    tr_i = 0
    te_i = 0
    for channel in channels:
        for window in range(num_windows):
            # Train set
            for patient in patients_train:

                X_train[tr_i, :], y_train[tr_i] = (
                    get_one_record_based_on_sorted_rr_intervals(data_df, vec_length, channel, window, patient))
                tr_i += 1

            # Test set
            for patient in patients_test:
                X_test[te_i, :], y_test[te_i] = (
                    get_one_record_based_on_sorted_rr_intervals(data_df, vec_length, channel, window, patient))
                te_i += 1

    return X_train, X_test, y_train, y_test


def get_one_record_based_on_sorted_rr_intervals(data_df, vec_length, channel, window, patient):
    # make one record here
    # for algorithm in algorithms:
    #     pass
    # indices = data_df.query('channel == 0 & window_num == 0 & patient == 209').index
    indices = data_df.query(f'channel == {channel} & window_num == {window} & patient == {patient}').index
    indices = np.sort(indices)

    num_algorithms = len(indices)
    x_vector = np.zeros(vec_length * num_algorithms)

    noise_label = data_df.loc[indices[0], 'noise_label']

    for index_i, index in enumerate(indices):
        try:
            sorted_rr_intervals = np.sort(data_df.loc[index, 'rr_intervals'])[::-1]
        except np.exceptions.AxisError:
            pass
            # print("numpy.exceptions.AxisError")  # for tqdm
            # print(data_df.loc[index, 'algorithm'])
            # print(data_df.loc[index, 'rr_intervals'])

        try:
            sorted_rr_intervals_length = len(sorted_rr_intervals)
        except UnboundLocalError:
            sorted_rr_intervals_length = 0

        if vec_length <= sorted_rr_intervals_length:
            temp = sorted_rr_intervals[:vec_length]
        else:
            temp = np.zeros(vec_length)
            if sorted_rr_intervals_length > 0:
                temp[:sorted_rr_intervals_length] = sorted_rr_intervals

        start_i = index_i * vec_length
        end_i = (index_i + 1) * vec_length
        x_vector[start_i:end_i] = temp

    return x_vector, noise_label

def build_ml_model(X_train, y_train):
    # train
    # clf_ert = ExtraTreesClassifier(n_estimators=100).fit(X_train, y_train)
    clf_rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    # clf_xgb = GradientBoostingClassifier(n_estimators=100).fit(X_train, y_train)
    # clf_gb = GradientBoostingClassifier(n_estimators=100).fit(X_train, y_train)
    # clf_xgb = XGBClassifier(n_jobs=1).fit(X_train, y_train)

    return clf_rf

def build_several_ml_model(X_train, y_train):
    # train
    clf_ert = ExtraTreesClassifier(n_estimators=100).fit(X_train, y_train)
    clf_rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    # clf_xgb = GradientBoostingClassifier(n_estimators=100).fit(X_train, y_train)
    clf_gb = GradientBoostingClassifier(n_estimators=100).fit(X_train, y_train)
    clf_xgb = XGBClassifier(n_jobs=1).fit(X_train, y_train)
    clf_dt = DecisionTreeClassifier().fit(X_train, y_train)
    clf_ls = make_pipeline(StandardScaler(), LinearSVC()).fit(X_train, y_train)
    clf_rbfsvm = make_pipeline(StandardScaler(), SVC()).fit(X_train, y_train)

    return clf_ert, clf_rf, clf_xgb, clf_dt, clf_ls, clf_rbfsvm


def evaluate(model, X_test, y_test, print_flag=False):
    # predict
    y_pred = model.predict(X_test)
    y_true = y_test

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1_performance = f1_score(y_true, y_pred)  # , average='weighted'
    acc_performance = accuracy_score(y_true, y_pred)
    precision_performance = precision_score(y_true, y_pred)
    recall_performance = recall_score(y_true, y_pred)

    if print_flag:
        print("tn, fp, fn, tp: ", tn, fp, fn, tp)
        print("f1_performance: ", f1_performance)
        print("acc_performance", acc_performance)
        print("precision_performance", precision_performance)
        print("recall_performance", recall_performance)

        print("\nclassification_report")
        print(classification_report(y_true, y_pred, target_names=['non noise', 'noise']))

    return f1_performance, acc_performance, precision_performance, recall_performance