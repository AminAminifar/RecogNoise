import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier

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
        sorted_rr_intervals = np.sort(data_df.loc[index, 'rr_intervals'])[::-1]
        sorted_rr_intervals_length = len(sorted_rr_intervals)
        if vec_length <= sorted_rr_intervals_length:
            temp = sorted_rr_intervals[:vec_length]
        else:
            temp = np.zeros(vec_length)
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

    return clf_rf


def evaluate(model, X_test, y_test):
    # predict
    y_pred = model.predict(X_test)
    y_true = y_test

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1_performance = f1_score(y_true, y_pred, average='weighted')
    acc_performance = accuracy_score(y_true, y_pred)
    print("tn, fp, fn, tp: ", tn, fp, fn, tp)
    print("f1_performance: ", f1_performance)
    print("acc_performance", acc_performance)

    print(classification_report(y_true, y_pred, target_names=['non noise', 'noise']))