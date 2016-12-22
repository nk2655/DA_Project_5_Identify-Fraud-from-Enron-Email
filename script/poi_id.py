# load data
import pandas as pd
data_dict = pd.read_pickle('pkl/final_project_dataset.pkl')

# Delete Outliers
data_dict.pop('TOTAL',0)
data_dict.pop('LOCKHART EUGENE E',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

# Create new features
def ratio_features(num,den):
    new_feature=[]
    for key in data_dict:
        if data_dict[key][num]!="NaN" or  data_dict[key][den]!="NaN":
            new_feature.append(float(data_dict[key][num])/float(data_dict[key][den]))
        else:
            new_feature.append(0.)
    return new_feature

ratio_from_poi=ratio_features('from_poi_to_this_person','to_messages')
ratio_to_poi=ratio_features('from_this_person_to_poi','from_messages')

for i,key in enumerate(data_dict):
    data_dict[key]["ratio_from_poi"]=ratio_from_poi[i]
    data_dict[key]["ratio_to_poi"]=ratio_to_poi[i]

new_features=['ratio_from_poi','ratio_to_poi']

# create Features list
financial_features=['salary', 'deferral_payments', 'total_payments',
 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
 'long_term_incentive', 'restricted_stock', 'director_fees'] 

email_features=['to_messages', 'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'shared_receipt_with_poi']

features_list = ['poi']+financial_features+email_features

features_list+=new_features

### repeat this section ######################################
# get best features by SelectKbest
# tested best value for K by hand, put 4 to 10 in K each time and caculated F1 with Decision Tree
# K=5 will get best result
import sys
sys.path.append('script/')
from feature_format import featureFormat, targetFeatureSplit
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=5)
selectedFeatures = selector.fit(features,labels)
best_features = [features_list[i] for i in selectedFeatures.get_support(indices=True)]
print 'Best features: ', best_features

# update labels and features
import numpy as np
np.random.seed(42)

data = featureFormat(data_dict, best_features)
labels, features = targetFeatureSplit(data)

    ## split data into training and testing
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

from sklearn.tree import DecisionTreeClassifier
from tester import test_classifier
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
test_classifier(clf, data_dict, best_features)

##############################################################
# Decision Tree
from time import time
t0 = time()
clf = DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
test_classifier(clf, data_dict, best_features)
print 'Running time: ', round(time()-t0, 3), 's'

# Tune Decision Tree
from sklearn.grid_search import GridSearchCV
t0 = time()	
param_grid = {
         'min_samples_split': [2, 3, 4, 5, 6, 7, 8],
          'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],
            'max_features': range(3,5)
          }
clf = GridSearchCV(DecisionTreeClassifier(), param_grid)
clf = clf.fit(features_train, labels_train)
print "Best estimator found by grid search:"
print clf.best_estimator_
print 'Running time: ', round(time()-t0, 3), 's'

# dump classifier and_data
from tester import dump_classifier_and_data
dump_classifier_and_data(clf, data_dict, best_features)