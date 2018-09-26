from easyautoml.ftutils import get_train_data, get_test_data
from easyautoml.tpotutils import tpot_train, tpot_score

# GLOBAL CONFIG
TPOT_MAX_TIME_MINS = 1
TPOT_MAX_EVAL_TIME_MINS = 0.04
TPOT_POPULATION_SIZE = 40
N_JOBS=1

def tpot_with_ft(
    project, 
    train_file_name, 
    test_file_name, 
    submit_file, 
    export_file, 
    prediction_target, 
    prediction_key, 
    prediction_type, 
    variable_types={}, 
    scoring_func=None, 
    predictProba=False, 
    predictInt=False, 
    getdummies=False, 
    drop_train_columns=None, 
    drop_score_columns=None):

    X, y = get_train_data(
        project=project, 
        train_file=train_file_name, 
        prediction_key=prediction_key, 
        prediction_target=prediction_target, 
        variable_types=variable_types, 
        drop_columns=drop_train_columns)

    tpot_instance = tpot_train(
        project=project, 
        X=X,
        y=y, 
        prediction_type=prediction_type, 
        export_file=export_file, 
        scoring_func=scoring_func, 
        max_time_mins=TPOT_MAX_TIME_MINS, 
        n_jobs=N_JOBS, 
        population_size=TPOT_POPULATION_SIZE)
    

    X_test, index_column = get_test_data(
        project=project, 
        testfile=test_file_name, 
        prediction_key=prediction_key, 
        prediction_target=prediction_target, 
        variable_types=variable_types, 
        drop_columns=drop_score_columns)


    tpot_score(
        tpot=tpot_instance, 
        project=project, 
        X=X_test, 
        index_column = index_column,
        prediction_target=prediction_target, 
        prediction_key=prediction_key, 
        submit_file=submit_file, 
        predictProba=predictProba, 
        predictInt=predictInt, 
        getdummies=getdummies)