import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  AUC = []
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = metrics.roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc
    AUC.append(roc_auc)
  return np.mean(AUC)

def get_performance_evaluation(y:np.ndarray=None, pred:np.ndarray=None):
    Accuracy = 100.0 * metrics.accuracy_score(y, pred)
    try:
        AUC = roc_auc_score_multiclass(y, pred)
    except:
        AUC = 0.0
    Recall = metrics.recall_score(y, pred, average='macro')
    Precision = metrics.precision_score(y, pred, average='macro')    
    CM = metrics.confusion_matrix(y, pred)
    
    GM = np.prod(np.diag(CM)) ** (1.0/CM.shape[0])
    return Accuracy, AUC, Precision, Recall, GM, CM






def cross_validation(model, X:np.ndarray=None, Y:np.ndarray=None, n_splits:int=3, seed:int=42, VERBOSE:bool=True):

    cv = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

    train_results = {'Accuracy':[], 'AUC':[], 'Precision':[], 'Recall':[], 'GM':[]}
    test_results = {'Accuracy':[], 'AUC':[], 'Precision':[], 'Recall':[], 'GM':[]}
    CM = None
    for i, (train, test) in enumerate(cv.split(X, Y)):
        # Train model
        if (type(model).__name__ == 'XGBClassifier'):
            # weights = [Y[train].shape[0] /np.where(Y[train] == i)[0].shape[0] for i in np.unique(Y[train])]

            model.fit(X[train], Y[train], 
                    eval_metric = 'auc', 
                    eval_set = [ (X[train], Y[train]), (X[test], Y[test]) ],
                    # sample_weight = [weights[int(x)] for x in Y[train]],
                    verbose = False);
        elif (type(model).__name__ == 'LGBMClassifier'):
            model.fit(X[train], Y[train], 
                    eval_metric = 'logloss', 
                    eval_set = [ (X[train], Y[train]), (X[test], Y[test]) ],
                    verbose = False);        
        else:
            model.fit(X[train], Y[train]) 
    
    
        # Evaluation on Training set
        pred = model.predict( X[train] )        
        Accuracy, AUC, Precision, Recall, GM, _ = get_performance_evaluation(Y[train], pred) 

        train_results['Accuracy'].append(Accuracy)
        train_results['AUC'].append(AUC)
        train_results['Precision'].append(Precision)
        train_results['Recall'].append(Recall)
        train_results['GM'].append(GM)


        # Evaluation on Testing set
        pred = model.predict( X[test] )        
        Accuracy, AUC, Precision, Recall, GM, CM_fold = get_performance_evaluation(Y[test], pred) 

        test_results['Accuracy'].append(Accuracy)
        test_results['AUC'].append(AUC)
        test_results['Precision'].append(Precision)
        test_results['Recall'].append(Recall)
        test_results['GM'].append(GM)

        if CM is None:
            CM = CM_fold
        else:
            CM += CM_fold     
    
    return train_results, test_results, CM



def single_run(model, trainX:np.ndarray=None, trainY:np.ndarray=None, testX:np.ndarray=None, testY:np.ndarray=None, VERBOSE:bool=True):

    train_results = {'Accuracy':None, 'AUC':None, 'Precision':None, 'Recall':None, 'GM':None}
    test_results = {'Accuracy':None, 'AUC':None, 'Precision':None, 'Recall':None, 'GM':None}
    
    # Train model
    if (type(model).__name__ == 'XGBClassifier'):
        # weights = [Y[train].shape[0] /np.where(Y[train] == i)[0].shape[0] for i in np.unique(Y[train])]

        model.fit(trainX, trainY, 
                eval_metric = 'auc', 
                eval_set = [ (trainX, trainY), (testX, testY) ],
                # sample_weight = [weights[int(x)] for x in trainY],
                verbose = False);
    elif (type(model).__name__ == 'LGBMClassifier'):
        model.fit(trainX, trainY, 
                    eval_metric = 'logloss', 
                    eval_set = [ (trainX, trainY), (testX, testY) ],
                    verbose = False);        
    else:
        model.fit(trainX, trainY) 
    
    
    # Evaluation on Training set
    pred = model.predict( trainX )        
    Accuracy, AUC, Precision, Recall, GM, _ = get_performance_evaluation(trainY, pred) 

    train_results['Accuracy'] = Accuracy
    train_results['AUC'] = AUC
    train_results['Precision'] = Precision
    train_results['Recall'] = Recall
    train_results['GM'] = GM


    # Evaluation on Testing set
    pred = model.predict( testX )        
    Accuracy, AUC, Precision, Recall, GM, CM = get_performance_evaluation(testY, pred) 

    test_results['Accuracy'] = Accuracy
    test_results['AUC'] = AUC
    test_results['Precision'] = Precision
    test_results['Recall'] = Recall
    test_results['GM'] = GM
  
    
    return model, train_results, test_results, CM, pred