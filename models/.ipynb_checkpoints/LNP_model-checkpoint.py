import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import operator, itertools, pickle, random, ast
import warnings
warnings.filterwarnings('ignore')

# Binary Classification using XGBoost

clf_VERSION = 'v1'
K = 5 # 5 fold cross validation

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,3))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.array(cm)
        cm = np.around(cm/cm.sum(axis=1)[:, None]*100).astype('int')
        print("Percentage confusion matrix")
        print(cm.sum(axis=1))
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return


def show_confusion_matrix(y, pred_array):
    y = np.array(y).astype(int)
    y_pred = np.array(pred_array)

    cnf_matrix = confusion_matrix(y, y_pred)
    np.set_printoptions(precision=2)
    sorted_cnf_matrix = cnf_matrix
    class_names = ['no', 'yes'] #[0, 1]

    plot_confusion_matrix(sorted_cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
    plt.show()
    return

def get_auc_plot(y, scores):
    y = np.array(y).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)

    # Compute ROC AUC and render the ROC curve. 
    accuracy_array = []
    sensitivity_array = [x*100 for x in tpr]
    specificity_array = [(1-x)*100 for x in fpr]#(1-fpr)*100
    for i, th in enumerate(thresholds):
        pred_array = []
        for s in scores:
            if s>th:
                pred_array.append(1)
            else:
                pred_array.append(0)
        accuracy_array.append(accuracy_score(y, pred_array))
    
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    for i in range(len(fpr)):
        if fpr[i] > 0.01:
            break
    return roc_auc, tpr[i], fpr[i]

def get_auc(y, scores):
    y = np.array(y).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    roc_auc = metrics.auc(fpr, tpr)

    for i in range(len(fpr)):
        if fpr[i] > 0.01:
            break
    return roc_auc, tpr[i], fpr[i]

def trainXGB(Xtrain, ytrain, Xtest, ytest, importance_array, column_names, hyperparameters, verbose):
    dtrain = xgb.DMatrix(Xtrain,label=ytrain)
    dtest = xgb.DMatrix(Xtest,label=ytest)
    print('Setting XGB params')
    evallist  = [(dtest,'test'), (dtrain,'train')]
    num_round = 220#60
    print('training the XGB classifier')
    bst = xgb.train(hyperparameters, dtrain, num_round, evallist, early_stopping_rounds=100, verbose_eval=False)
    #print('training completed, printing the relative importance: \
    #      (feature id: importance value)')
    importance = bst.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    
    # we will print from df1 dataframe, getting the corresponding feature names. 
    df1 = pd.DataFrame(importance, columns=['feature', 'fscore'])
    # Normalizing the feature scores
    df1['fscore'] = df1['fscore'] / df1['fscore'].sum()
    
    # adding a column of feature name
    print('check size of col names: ',len(column_names))
    df1['feature_names'] = pd.Series([column_names[int(f[0].replace("f", ""))] for f in importance])
    
    #Take Top 30
    importance_array = df1.copy()
    df1 = df1.nlargest(30, 'fscore')
    if verbose:
        df1.plot()
        df1.sort_values(by='fscore',ascending=True).plot(kind='barh', x='feature_names', y='fscore', legend=False, figsize=(6, 10))
        plt.title('XGBoost Feature Importance (Top 30)')
        plt.xlabel('Relative importance')
        plt.gcf().savefig('feature_importance_xgb.png')
        plt.show()

    return bst, importance_array


def classifier_train(X, y, runXGB, Xtest, ytest, pca_comp, smote, importance_array, column_names, hyperparameters, verbose): # pca = 0 means no PCA applied
    print('Normalising the input data...')
    scaler = StandardScaler()
    scaler.fit(X)  
    scaledX = scaler.transform(X)
    if smote == 1:
        X_resampled, y_resampled = SMOTE().fit_resample(np.array(scaledX), y.astype(int))
        #(SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, SMOTENC, KMeansSMOTE, RandomOverSampler, SMOTEENN, SMOTETomek)
        scaledX = X_resampled#pd.DataFrame(X_resampled, columns = data.columns)
        y = y_resampled #pd.DataFrame(pd.Series(y_resampled, name='label'))
    if pca_comp != 0:
        pca = PCA(n_components = pca_comp)
        pca.fit(scaledX)
        pca_scaledX = pca.transform(scaledX)
    else:
        pca_scaledX = scaledX
        pca =0

    if runXGB == 1:
        print('Running the XGB classifier')
        clf, importance_array = trainXGB(pca_scaledX, y, scaler.transform(Xtest), ytest, importance_array, column_names, hyperparameters, verbose)
        index = 1
    return pca_scaledX, y, scaler, pca, clf, index, importance_array, column_names


def classifier_test(scaledX, y, clf, index, scaler, pca, verbose):
    pca_scaledX = scaledX
    if pca != 0:
        pca_scaledX = pca.transform(scaledX)
    if index==1:# XGB
        pca_scaledXG = xgb.DMatrix(pca_scaledX, label=y)
        pred_array = clf.predict(pca_scaledXG, ntree_limit=clf.best_iteration)
        scores = pred_array

    if verbose:
        auc = get_auc_plot(y, scores)
    else:
        auc = get_auc(y, scores)
    # Compute confusion matrix
#    show_confusion_matrix(y, pred_array)
    return pred_array, auc, clf# error

def feature_selection(Xtrain, Xtest, importance_array, column_names, verbose):
    imp_vals = importance_array
    imp_vals['feature'] = imp_vals['feature'].map(lambda k: k.replace("f",""))
    imp_vals['feature'] = imp_vals['feature'].astype(int)
    imp_vals['fscore'] = imp_vals['fscore'].astype(float)
    imp_vals = imp_vals.sort_values('feature')
    if verbose:
        print('---importance values---')
        print('max:',np.max(imp_vals.iloc[:,1]))
        print('min:',np.min(imp_vals.iloc[:,1]))
        print('mean:',np.mean(imp_vals.iloc[:,1]))
    #threshold importance
    threshold = 0 #threshold on fscore sum of all 5 folds 0.007, 
    imp_vals = imp_vals.loc[imp_vals['fscore'] > threshold] #threshold importance
    filt = imp_vals.to_numpy()[:,0].astype(int)
    
    #filter both training and testing data
    column_names = column_names[filt]
    Xtrain_filt = np.array(Xtrain)[:, filt]
    Xtest_filt = np.array(Xtest)[:, filt]
    
    if verbose:
        print('Before Feat Sel: ',Xtrain.shape[1],' features')
        print('After Feat Sel: ',Xtrain_filt.shape[1],' features')
        print('------------------------')
    return filt, Xtrain_filt, Xtest_filt, column_names
    
def pandas_classifier(df, runXGB, K, importance_array, hyperparameters, verbose = True):
    print('Performing ' + str(K) + '-fold cross validation')
    auc_fold = []
    acc_fold = []
    for k in range(K):# performing K fold validation
        #if k == 0: # running only for k'th fold
            print('Fold_num = ' + str(k))
            training_rows = [i for i in range(len(df)) if i%K!=k]
            datatrain = df.loc[training_rows] # training
            testing_rows = [i for i in range(len(df)) if i%K==k]
            datatest = df.loc[testing_rows] # taking every k'th example for test
            Xtrain = datatrain.iloc[:, 0:-1]
            ytrain = datatrain.iloc[:, -1]
            Xtest = datatest.iloc[:, 0:-1]
            ytest = datatest.iloc[:, -1]
            print('--------------------------------------------------------------')
            print('Calling the classifier to train')
            importance_array = pd.DataFrame()
            Xtrain_scaled, ytrain, scaler, pca, clf, index, importance_array, column_names = classifier_train(Xtrain, ytrain, runXGB, Xtest, ytest, 0, 1, importance_array, Xtrain.columns, hyperparameters, verbose)
            print('Feature selection using training feature importance')
            Xtest_scaled = scaler.transform(Xtest)
            imp_features, Xtrain_scaled, Xtest_scaled, column_names = feature_selection(Xtrain_scaled, Xtest_scaled, importance_array, column_names, verbose)
            print('Repeat training on filtered training data')
            importance_array = pd.DataFrame()
            clf, importance_array = trainXGB(Xtrain_scaled, ytrain, Xtest_scaled, ytest, importance_array, column_names, hyperparameters, verbose)
            print('Analysing the test predictions for fold num ', k)
            pred_array, auc, clf = classifier_test(Xtest_scaled, ytest, clf, index, scaler, 0, verbose)
            auc_fold.append(auc[0])
            print('test auc = '+str(auc[0]) )
            accuracy = accuracy_score(ytest.astype(int), np.round(pred_array))
            acc_fold.append(accuracy)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            if verbose:
                show_confusion_matrix(ytest, np.round(pred_array))
            print('------------------------------------------------------------')
    if K != 0:
        print('************************************************************************')
        print(auc_fold)#, sum(np.array(auc_fold))/int(K))
        print('Average '+str(K)+' fold CV AUC= ', str(sum(np.array(auc_fold))/int(K)))
        print('Average '+str(K)+' fold CV Accuracy= ', str(sum(np.array(acc_fold))/int(K)))
        print('************************************************************************')
    print('Training the classifier on complete train dataset to get IMP features')
    Xtrain_full, ytrain_full, scaler, pca, clf, index, importance_array, column_names = classifier_train(df.iloc[:, 0:-1], df.iloc[:, -1],
                                               runXGB, df.iloc[:, 0:-1], df.iloc[:, -1], 0, 1, importance_array, df.iloc[:, 0:-1].columns, hyperparameters, verbose)
    return importance_array, sum(np.array(auc_fold))/int(K), sum(np.array(auc_fold))/int(K)


def train_model(X_train, y_train, X_val, y_val, random_search_params):
    #Parameters
    np.random.seed(15)
    random.seed(15)
    importance_array = pd.DataFrame()
    runXGB = 1
    pca = 0 #broken
    smote = 1
    verbose = False

    #Training
    X_train, y_train, scaler, pca, clf, index, importance_array, column_names = classifier_train(X_train, y_train, runXGB, X_val, y_val, pca, smote, importance_array, X_train.columns, random_search_params, verbose)
    print('Feature selection using training feature importance')
    X_val = scaler.transform(X_val)
    imp_features, X_train, X_val, column_names = feature_selection(X_train, X_val, importance_array, column_names, verbose)
    print('Repeat training on filtered training data')
    importance_array = pd.DataFrame()
    clf, importance_array = trainXGB(X_train, y_train, X_val, y_val, importance_array, column_names, random_search_params, verbose)
    return clf, index, scaler, pca, imp_features, verbose
                
def test_model(X_test, y_test, clf, index, scaler, pca, imp_features, verbose):
    #Testing
    print('Analysing the test predictions')
    X_test = scaler.transform(X_test)
    X_test = np.array(X_test)[:, imp_features]
    pred_array, auc, clf = classifier_test(X_test, y_test, clf, index, scaler, pca, verbose)

    #Evaluation
    print('test auc = '+str(auc[0]) )
    accuracy = accuracy_score(y_test.astype(int), np.round(pred_array))
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    show_confusion_matrix(y_test, np.round(pred_array))

def main():

    #Load data
    model_name = 'RNA' #'mRNA', 'siRNA' 

    data_train = pd.read_csv('../datasets/'+model_name+'_train.csv')
    data_test = pd.read_csv('../datasets/'+model_name+'_test.csv')

    #Drop Payload if not RNA model (RNA model contains both mRNA and siRNA nanoparticles)
    # data_train = data_train.drop(['Payload'],1)
    # data_test = data_test.drop(['Payload'],1)
        
    #Separate X and y
    y_train = np.array(data_train['label'])
    X_train = data_train.drop(['label','Lipomer', 'Cholesterol', 'HelperLipid', 'PEGChain', 'PEG MW', 'diameter'], 1)
    y_test = np.array(data_test['label'])
    X_test = data_test.drop(['label','Lipomer', 'Cholesterol', 'HelperLipid', 'PEGChain', 'PEG MW', 'diameter'], 1)


    #Load Parameters
    name = model_name
    random_results = pd.read_csv('./params/RS_results_'+name+'.csv')

    with open('./params/hyperparameters'+name+'.pkl', 'rb') as f:
        random_search_params = pickle.load(f)

    #Select different parameters
    rank = 0
    random_search_params = ast.literal_eval(random_results.loc[rank, 'params'])

    # Validate Model

    # Best hyperparameters from random search
    np.random.seed(15)
    random.seed(15)
    df = pd.concat([X_train, pd.DataFrame(pd.Series(y_train, name='label'))], axis=1)
    importance_array = pd.DataFrame()
    verbose = False
    importance_array, auc, accuracy = pandas_classifier(df, 1, 5, importance_array, random_search_params, verbose)


    # Train Model
    clf, index, scaler, pca, imp_features, verbose  = train_model(X_train, y_train, X_test, y_test, random_search_params)


    # Test Model
    test_model(X_test, y_test, clf, index, scaler, pca, imp_features, verbose)

if __name__ == '__main__':
    main()