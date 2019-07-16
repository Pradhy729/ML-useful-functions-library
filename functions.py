def ConvertColumnToCategorical(df_in, column_names):
    for column in column_names:
        df_in.loc[:,column] = pd.cut(df_in[column],4, labels=False, duplicates='drop').astype('int')

def remove_correlated_features(df):
    corr_matrix = df.corr().abs()
    upper = np.triu(corr_matrix,1)
    upper_df = pd.DataFrame(upper, columns=corr_matrix.columns)
    to_drop = [column for column in upper_df.columns if any(abs(upper_df[column]) > 0.85)]
    df.drop(to_drop, axis=1, inplace=True

def Imbalanced_dataSampling(data_in, target_in, sample_type='over-sample' ,sample_algorithm='smote'):
    if sample_type=='over-sample':
        if sample_algorithm == 'smote':
            from imblearn.over_sampling import BorderlineSMOTE
            imb_sampler = BorderlineSMOTE(sampling_strategy='not majority', n_jobs=-1)
        elif sample_algorithm == 'random':
            from imblearn.over_sampling import RandomOverSampler
            imb_sampler = RandomOverSampler(sampling_strategy='not majority', n_jobs=-1)
    elif sample_type=='under-sample': 
        from imblearn.under_sampling import RandomUnderSampler
        imb_sampler = RandomUnderSampler(sampling_strategy={0:nPtPerCat, 2:nPtPerCat})
    elif sample_type=='combine':
        from imblearn.combine import SMOTETomek
        imb_sampler = SMOTETomek(sampling_strategy=dict(zip(range(5),np.ones(5)*nPtPerCat)))
    
    data_X_res, data_Y_res = imb_sampler.fit_resample(data_in,target_in)
    print('Balanced data:')
    print('Class\tCount')
    for i in np.unique(data_Y_res):    
        print(str(i) + '\t' + str(np.bincount(data_Y_res)[i]))
    return data_X_res, data_Y_res

def adaboost_randomized_search(Xtrain, Ytrain)
    dtc_clf = DecisionTreeClassifier()
    adaboost_clf = AdaBoostClassifier(base_estimator=dtc_clf)
    learning_rates = [x for x in np.linspace(0.01, 0.5, 3)]
    # Number of trees 
    n_estimators = [int(x) for x in [10, 20, 30]]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(3, 20, 3)]
    max_depth.append(None)
    #Min sample for splitting a node
    min_sample_splits = [int(x) for x in np.linspace(2,100,10)]
    # Create the random grid
    random_grid = {'learning_rate': learning_rates,
                 'n_estimators': n_estimators,
                 'base_estimator__max_depth': max_depth }

    adaboost_clf_reg = RandomizedSearchCV(estimator = adaboost_clf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, scoring = 'f1_weighted', n_jobs = -1).fit(Xtrain,Ytrain)

def xgboost_randomized_search(Xtrain, Ytrain):
    xgb = XGBClassifier(nthreads=-1)

    learning_rates_xgb = [x for x in np.linspace(0.01, 0.5, 3)]

    max_depth = [int(x) for x in np.linspace(3, 10, 5)]
    min_child_weight = [x for x in [1,2,3]]
    n_estimators = [int(x) for x in np.arange(30, 100, 10)]


    random_grid = {'learning_rate': learning_rates_xgb,
               'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_child_weight': min_child_weight,
               'gamma': [x for x in np.arange(0,20,2)]}

    xgb_reg = RandomizedSearchCV(xgb,param_distributions=random_grid, scoring='f1_weighted', cv=3, n_iter=10, verbose=2, n_jobs=-1).fit(Xtrain,Ytrain)
