import easyautoml
import featuretools as ft
from subprocess import run


def main():
    PROJECT = "titanic"
    DATA_DIR = "data/" + PROJECT + "/"
    PREDICTION_TARGET = 'Survived'
    PREDICTION_KEY = 'PassengerId'
    PREDICTION_TYPE = 'classification'
    TRAIN_DATA_FILE = DATA_DIR+"train.csv"
    TEST_DATA_FILE = DATA_DIR+"test.csv"

    # following should not need to be modified
    SUBMIT_FILE = DATA_DIR + PROJECT + "_submission_tpot.csv"
    TRAIN_VECTOR = DATA_DIR + PROJECT + "_train.json"
    TRAIN_VECTOR_CSV = DATA_DIR + PROJECT + "_train.csv"
    SCORE_VECTOR = DATA_DIR + PROJECT + "_score.json"
    SCORE_VECTOR_CSV = DATA_DIR + PROJECT + "_score.csv"
    TPOT_EXPORT = DATA_DIR + PROJECT + "_tpot_pipeline.py"

    SUBMIT_FILE_AUTOML = DATA_DIR + PROJECT + "_submission_automl.csv"

    TPOT_EXPORT_FT = DATA_DIR + PROJECT + "_tpot_pipeline_ft.py"
    SUBMIT_FILE_FT = DATA_DIR + PROJECT + "_submission_tpot_ft.csv"

    VAR_TYPES = {
        'Pclass':ft.variable_types.Categorical,
        'Name':ft.variable_types.Text,
        'Ticket':ft.variable_types.Text,
        'Cabin':ft.variable_types.Text
    }

    #run(["kaggle", "competitions", "download", "-c", PROJECT, "-p", DATA_DIR])

    easyautoml.utils.tpot_with_ft(
        project=PROJECT,
        train_file_name=TRAIN_DATA_FILE,
        test_file_name=TEST_DATA_FILE,
        variable_types=VAR_TYPES,
        submit_file=SUBMIT_FILE_FT,
        export_file=TPOT_EXPORT_FT,
        prediction_target=PREDICTION_TARGET,
        prediction_key=PREDICTION_KEY,
        prediction_type=PREDICTION_TYPE)

    #run(["kaggle", "competitions", "submit", PROJECT, "-f", SUBMIT_FILE_FT, "-m", "tpot with ft"])

    #run(["kaggle", "competitions", "submissions", PROJECT])