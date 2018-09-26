# Import required libraries
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import make_scorer
import numpy as np
import pandas as pd
import json


# define the msele
# Make a custom metric function
def msele(real, predicted):
    '''msele '''
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5

# Make a custom a scorer from the custom metric function
# Note: greater_is_better=False in make_scorer below would mean that the scoring function should be minimized.
msele_scorer = make_scorer(msele, greater_is_better=False)

def logloss(true_label, predicted, eps=1e-15):
    p = np.clip(predicted, eps, 1 - eps)
    if true_label == 1:
        return -np.log(p)
    else:
        return -np.log(1 - p)

from sklearn.metrics import log_loss

logloss_scorer = make_scorer(log_loss, greater_is_better=False)

    
def tpot_train(
    project, 
    X, 
    y, 
    export_file, 
    prediction_type, 
    train_size=0.75, 
    max_time_mins=1, 
    max_eval_time_mins=0.04, 
    population_size=40, 
    scoring_func=None, 
    n_jobs=1):

    print("==========train / test split for training size {}".format(train_size))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
    print(X_train.shape, y_train.shape)
    
    print("==========Start training the model...")
    print("==========max_time_mins: {}".format(max_time_mins))
    print("==========max_eval_time_mins: {}".format(max_eval_time_mins))
    print("==========population_size: {}".format(population_size))
    print("==========n_jobs: {}".format(n_jobs))
    
    # predition type:
    # - regression
    # - classification
    if (prediction_type == "classification"):
        tpot = TPOTClassifier(verbosity=2, max_time_mins=max_time_mins, max_eval_time_mins=max_eval_time_mins, population_size=population_size, scoring=scoring_func, n_jobs=n_jobs, early_stop=3)
    else:
        tpot = TPOTRegressor(verbosity=2, max_time_mins=max_time_mins, max_eval_time_mins=max_eval_time_mins, population_size=population_size, scoring=scoring_func, n_jobs=n_jobs, early_stop=3)
        
    tpot.fit(X_train, y_train)
    
    try:
        holdout_score = tpot.score(X_test, y_test)
        print("==========holdout set score is {}".format(holdout_score))
    except:
        print("==========Unexpected error when score holdout set")
   
    print("==========export tpot to {}".format(export_file))
    tpot.export(export_file)
    
    return tpot

    
def tpot_score(
    tpot, 
    project, 
    X, 
    index_column,
    submit_file, 
    prediction_target, 
    prediction_key, 
    predictProba=False, 
    predictInt=False, 
    getdummies=False):

    print("==========Start scoring...")
    
    # Generate the predictions
    if predictProba:
        submission = tpot.predict_proba(X)[:,1]
    else:
        submission = tpot.predict(X)
        
    if predictInt:
        submission = np.clip(submission.astype('int'),0,None)

    # Create the submission file# Create 
    if prediction_key is not None:
        if getdummies:
            final = pd.concat([index_column, pd.get_dummies(submission)], axis=1)
        else:
            final = pd.DataFrame({prediction_key: index_column, prediction_target: submission})
        
        final.to_csv(submit_file, index = False)
    else:
        final = pd.DataFrame({prediction_target: submission})
        final.to_csv(submit_file, index = True)

    print("==========Submission shape {}".format(final.shape))


# def tpot_with_ft(
#     project, 
#     train_file_name, 
#     test_file_name, 
#     submit_file, 
#     export_file, 
#     prediction_target, 
#     prediction_key, 
#     prediction_type, 
#     variable_types={}, 
#     scoring_func=None, 
#     predictProba=False, 
#     predictInt=False, 
#     getdummies=False, 
#     drop_train_columns=None, 
#     drop_score_columns=None):

#     X, y = ftutils.get_train_data(
#         project=project, 
#         train_file=train_file_name, 
#         prediction_key=prediction_key, 
#         prediction_target=prediction_target, 
#         variable_types=variable_types, 
#         drop_columns=drop_train_columns)

#     tpot_instance = tpot_train(
#         project=project, 
#         X=X,
#         y=y, 
#         prediction_type=prediction_type, 
#         export_file=export_file, 
#         scoring_func=scoring_func, 
#         max_time_mins=TPOT_MAX_TIME_MINS, 
#         n_jobs=N_JOBS, 
#         population_size=TPOT_POPULATION_SIZE)
    

#     X_test, index_column = ftutils.get_test_data(
#         project=project, 
#         testfile=test_file_name, 
#         prediction_key=prediction_key, 
#         prediction_target=prediction_target, 
#         variable_types=variable_types, 
#         drop_columns=drop_score_columns)


#     tpot_score(
#         tpot=tpot_instance, 
#         project=project, 
#         X=X_test, 
#         index_column = index_column,
#         prediction_target=prediction_target, 
#         prediction_key=prediction_key, 
#         submit_file=submit_file, 
#         predictProba=predictProba, 
#         predictInt=predictInt, 
#         getdummies=getdummies)

