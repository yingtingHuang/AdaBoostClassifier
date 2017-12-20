import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''
    classifierArr=[]
    Alpha=[]
        
    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        
        X = np.mat(X) # ndarray -> matrix
        y = np.mat(y) # ndarray -> matrix
        n_samples ,n_features= np.shape(X)
        D = 1/n_samples * np.ones((n_samples,), dtype=np.int)      #init D to all equal      
        for i in range(self.n_weakers_limit):
            # Training a base classifier 
            clf = DecisionTreeClassifier(max_features=n_features ,max_depth=2) 
            # Build a decision tree classifier from the training set (X, y).
            clf = clf.fit(X,y,sample_weight=D)   
            # Predict class or regression value for X.
            classEst = clf.predict(X)      
            classEst= classEst.reshape(n_samples,1)
            # Calculate the classification error rate of the base classifier on the training set. 
            errArr = np.mat(np.ones((n_samples,1)))
            errArr[classEst == y] = 0               
            error =  D.T*errArr
            # Calculate the parameter according to the classification error rate . 
            alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
            # Store Params in Array
            self.classifierArr.append(clf)               
            self.Alpha.append(alpha)
            # exponent for D calc, getting messy 
            expon = np.multiply(-1*alpha* np.mat(y),classEst) 
            # Calculate New weight D for next iteration
            D = np.multiply(D,np.exp(expon).T)                                 
            D = np.mat(D/D.sum())
            D = D.flatten().A[0] # mat->1D array by using flatten           
            # Predict the catagories for the training samples
            y_pred = self.predict( X, threshold=0) 
            # Calculate training error of all classifiers, 
            aggErrors = np.multiply( y_pred != np.mat(y),np.ones((n_samples,1)))  
            errorRate = aggErrors.sum()/n_samples
            print ("total error: ",errorRate)
            # if the training error is 0 quit for loop early (use break)
            if errorRate == 0.0: 
                break
        
        return 
        

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        
        n_samples = np.shape(X)[0]
        aggClassEst = np.mat(np.zeros((n_samples,1)))
        for i in range(len(self.classifierArr)):
            # get the classifier i
            clf = self.classifierArr[i]
            # Predict class or regression value for X.
            classEst = clf.predict(X)       
            classEst= classEst.reshape(n_samples,1)
            # Calculate the weighted sum score of the whole base classifiers for given samples
            aggClassEst += self.Alpha[i]*classEst
            
        return aggClassEst

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        
        X = np.mat(X)  # ndarray -> matrix
        # Calculate the weighted sum score of the whole base classifiers for given samples.
        aggClassEst = self.predict_scores(X)       
        # mark the sample whose predict scores greater than the threshold as positive, 
        # on the contrary as negative.
        y_pred = np.zeros_like(aggClassEst)
        y_pred[aggClassEst> threshold] =  1
        y_pred[aggClassEst<=threshold] = -1
        
        return y_pred

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
