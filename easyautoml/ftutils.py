# feature tools
import featuretools as ft
import pandas as pd

def get_ft_entities(es, project, prediction_key, data, variable_types):
    
    if prediction_key is not None:
        ft_entities = es.entity_from_dataframe(
            entity_id=project,
            index=prediction_key,
            dataframe=data,
            variable_types=variable_types)
    else:
        ft_entities = es.entity_from_dataframe(
            entity_id=project,
            index='ImageId',
            dataframe=data,
            make_index=True,
            variable_types=variable_types)
    
    return ft_entities

def get_train_data(
    project, 
    train_file, 
    prediction_key, 
    prediction_target, 
    variable_types={}, 
    drop_columns=None):
    
    # Read the training data
    print("==========Reading the training file {}".format(train_file))
    train_data = pd.read_csv(train_file)
    train_data.head(5)
    
    print("==========Preparing training labels for target {}".format(prediction_target))
    train_labels = train_data[prediction_target].values
    train_data = train_data.drop(prediction_target, axis=1)
    
    if drop_columns is not None:
        print("==========dropping columns {}".format(drop_columns))
        train_data = train_data.drop(drop_columns, axis=1)
    
    print("==========Generating the feature with featuretools")
    
    es = ft.EntitySet(project)
    
    entities = get_ft_entities(
        es=es,
        project=project,
        prediction_key=prediction_key,
        data=train_data,
        variable_types=variable_types
    )
    
    print("==========entities are:")
    print(entities)
    
    feature_matrix, feature_defs = ft.dfs(
        entityset=entities,
        target_entity=project
    )


    feature_matrix_enc, features_enc = ft.encode_features(feature_matrix, feature_defs)
    print("==========columns are:")
    print(feature_matrix_enc.columns)
    
    
    print("==========features_enc are:")
    print(features_enc)
    
    
    print("==========saving features to {}".format(project))
    ft.save_features(features_enc, "data/{}/ft_features".format(project))

    return feature_matrix_enc, train_labels

def get_test_data(
    project, 
    testfile, 
    prediction_key, 
    prediction_target, 
    variable_types={},
    drop_columns=None):

    print("==========Reading test data file {}".format(testfile))
    test_data = pd.read_csv(testfile)
    print(test_data.describe())
    
    if drop_columns is not None:
        print("==========dropping columns {}".format(drop_columns))
        test_data = test_data.drop(drop_columns, axis=1)
    
    es = ft.EntitySet(project)
    
    entities = get_ft_entities(
        es=es,
        project=project,
        prediction_key=prediction_key,
        data=test_data,
        variable_types=variable_types
    )
    

    print("==========entities are:")
    print(entities)
    
    print("==========Reading features from {}".format(project))
    saved_features = ft.load_features("data/{}/ft_features".format(project))
    feature_matrix = ft.calculate_feature_matrix(saved_features, entities)
    index_column = test_data[prediction_key]

    return feature_matrix, index_column