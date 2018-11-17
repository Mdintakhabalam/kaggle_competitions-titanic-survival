############################# TITANIC SURVIVAL #############################

# Importing library to manipulate with data
import numpy as np
import pandas as pd

# Importing library for visulization of data
import matplotlib.pyplot as plt
import seaborn as sns

# Importing library for machine learning algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

############### Data Manipulation ################

# We know that in data science we have many missing and dirty data and we need to deal with them..
# Importing data set
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()

# save passenger id for final submission
PassengerId = test.PassengerId

# merging train and test data
train_test = train.append(test, ignore_index = True)

# create indexes to separate data later on
train_idx = len(train)
test_idx = len(train_test) - len(test)
print(train_idx)
print(test_idx)

train_test.info()
train_test.head(5)

# From looking our dataset we can find out that we can just drop out the PassengerId, as it is not 
# that important for our machine learning model...
del train_test["PassengerId"]

# Now I want to extract the title's of each person name and group them according to their title..
# This will allow us to more accurately estimate the other features in the next few steps..
train_test['Title'] = train_test.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

# view the newly created feature
train_test.head()

# show count of titles
print("There are {} unique titles.".format(train_test.Title.nunique()))  
# show unique titles
print("\n", train_test.Title.unique())

# here we see there are 18 unique titles.. But now i want to normalize the titles as we see there some
# titles with same meaning but in deffeerent language....

# normalize the titles
normalized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}

# map the normalized titles to the current titles
train_test.Title = train_test.Title.map(normalized_titles)

# view value counts for the normalized titles
print(train_test.Title.value_counts())

# we will group the data by Sex, Pclass, and Title and then view the median age for the grouped classes.
grouped = train_test.groupby(['Sex','Pclass', 'Title'])
grouped.Age.median()

# apply the grouped median value on the Age NaN
train_test.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))

# view changes
train_test.info()

# fill Cabin NaN with U for unknown
train_test.Cabin = train_test.Cabin.fillna('U')

# find most frequent Embarked value and store in variable
most_embarked = train_test.Embarked.value_counts().index[0]

# fill NaN with most_embarked value
train_test.Embarked = train_test.Embarked.fillna(most_embarked)

# fill NaN with median fare
train_test.Fare = train_test.Fare.fillna(train_test.Fare.median())

# view changes
train_test.info()

# Now we have full neat and clean dataset to perform out next steps i.e. without NANs

# view the percentage of those that survived vs. those that died in the Titanic
train_test.Survived.value_counts(normalize=True)
# Looks like only 38% people survived ...

# Let's dig a little dipper and see the survival chance of people by sex..
group_sex = train_test.groupby('Sex')

# survival rates by class and sex
group_sex.Survived.mean()
# Here we see the survival chance of female are 74% and that of male are only 18%...

# Let's break it down by both passenger class and sex..
gp_Pcl_sex = train_test.groupby(['Pclass', 'Sex'])
gp_Pcl_sex.Survived.mean()
# It appears that 1st class female had 96% chance of survival , 1st class male had 36% chances of survival,
# 2nd class female had 92% chance of survival , 2nd class male had 15% chance of survival, 3rd class female
# had 50% chance of survivaal , and 3rd class male had only 13% chance of survivval.

# plot by Survivded, Sex
sns.factorplot(x='Sex', col='Survived', data=train_test, kind='count')

# plot by Pclass, Sex, Survived
sns.factorplot(x='Pclass', hue='Sex', col='Survived', data=train_test, kind='count')
# Here we see the social status played an important role for the survival of the people on titanic...

# see the status of all the features
train_test.describe()


# map first letter of cabin to itself
train_test.Cabin = train_test.Cabin.map(lambda x: x[0])

######## Feature Enginering ##########
# First we convert some categorical data into some numbers..
train_test.head()

# Convert the male and female groups to integer form
train_test.Sex = train_test.Sex.map({"male": 0, "female":1})

# create dummy variables for categorical features
pclass_dummies = pd.get_dummies(train_test.Pclass, prefix="Pclass")
title_dummies = pd.get_dummies(train_test.Title, prefix="Title")
cabin_dummies = pd.get_dummies(train_test.Cabin, prefix="Cabin")
embarked_dummies = pd.get_dummies(train_test.Embarked, prefix="Embarked")

# concatenate dummy columns with main dataset
train_test_dummies = pd.concat([train_test, pclass_dummies, title_dummies, cabin_dummies, embarked_dummies], axis=1)

# drop categorical fields
train_test_dummies.drop(['Pclass', 'Title', 'Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

train_test_dummies.head()

# create train and test data
train = train_test_dummies[ :train_idx]
test = train_test_dummies[test_idx: ]

# convert Survived back to int
train.Survived = train.Survived.astype(int)
train.head()
       
       
############################# Modeling ########################
# create X and y for data and target values
X = train.drop('Survived', axis=1).values
y = train.Survived.values       
       
# create array for test set
X_test = test.drop('Survived', axis=1).values       
       
###### RANDOM FOREST MODEL #############       


# instantiate Random Forest model
classifier = RandomForestClassifier(n_estimators=100,random_state=0,
                                    max_depth=10,min_samples_split=10,min_samples_leaf=10)

# Splitting the data into training and test set so that I can visulise my accuracy myself..
x_tr, x_te, y_tr, y_te= train_test_split(X,y)

# build and fit model
classifier.fit(x_tr,y_tr)
     
# random forrest prediction on test set      
print( classifier.score(x_te, y_te))
print( classifier.score(x_tr, y_tr))
y_pred = classifier.predict(X_test) 
      
#### Prepearing Gender_submission.csv file for kaggle submission
gender_submission = pd.DataFrame( {'PassengerId': PassengerId, 'Survived': y_pred} )       
gender_submission.to_csv('gender_submission.csv', index = False)      
       
       

