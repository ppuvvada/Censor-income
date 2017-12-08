
import pandas
import numpy
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, SelectFpr, f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier



class FeatureSelection(object):

    def runUnivariateSelectionMethod(self, X, y, score_func, k_val):
        algo = None
        if score_func == "chi2":
            algo = SelectKBest(score_func=chi2, k=k_val)
        else:
            algo = SelectKBest(score_func=f_classif, k=k_val)
        fit = algo.fit(X, y)

        list_of_selected = [zero_based_index for zero_based_index in list(algo.get_support(indices=True))]
        
        return fit, list_of_selected

    def runUnivariateSelectPercentileMethod(self, X, y, score_func, perc):
        algo = None
        if score_func == "chi2":
            algo = SelectPercentile(score_func=chi2, percentile=perc)
        else:
            algo = SelectPercentile(score_func=f_classif, percentile=perc)
        fit = algo.fit(X, y)

        list_of_selected = [zero_based_index for zero_based_index in list(algo.get_support(indices=True))]
        
        return fit, list_of_selected

    def runRFESelectionMethod(self, X, y, num_of_features):
        model = LogisticRegression()
        rfe = RFE(model, num_of_features)
        fit = rfe.fit(X, y)
        return fit
    
    def runRFESelectionMethodRF(self, X, y, num_of_features):
        model = RandomForestClassifier()
        rfe = RFE(model, num_of_features)
        fit = rfe.fit(X, y)
        return fit
    
    def runRFESelectionMethodSVM(self, X, y, num_of_features):
        model = SVR(kernel="linear")
        rfe = RFE(model, num_of_features)
        fit = rfe.fit(X, y)
        return fit
    
    def runRFESelectionMethodNB(self, X, y, num_of_features):
        model = GaussianNB()
        rfe = RFE(model, num_of_features)
        fit = rfe.fit(X, y)
        return fit
    
    def runRFESelectionMethodDT(self, X, y, num_of_features):
        model = DecisionTreeClassifier()
        rfe = RFE(model, num_of_features)
        fit = rfe.fit(X, y)
        return fit
    
    def runRFESelectionMethodKNN(self, X, y, num_of_features):
        model = KNeighborsClassifier()
        rfe = RFE(model, num_of_features)
        fit = rfe.fit(X, y)
        return fit

    def runTreesClassifierMethod(self, X, y):
        
        # feature extraction
        model = ExtraTreesClassifier()
        model.fit(X, y)

        return model.feature_importances_