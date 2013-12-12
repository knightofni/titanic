""" Writing my first randomforest code.
Author : knightofni
Date : 23rd September, 2012
please see packages.python.org/milk/randomforests.html for more

"""
import pandas as pd
import numpy as np
import pylab as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

numbins = 20


print 'Reading '
#Load in the training csv file
train_data = pd.read_csv('Data\\train.csv')
test_data = pd.read_csv('Data\\test.csv')
test_data.insert(1, 'Survived', 0)

data = [train_data, test_data]
cols = ['Survived', 'PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

print 'Processing '
for df in data:
    #I need to convert all strings to integer classifiers:
    #Male = 1, female = 0
    df.loc[df.Sex == 'female', 'Sex'] = 0
    df.loc[df.Sex == 'male', 'Sex'] = 1
    df.Sex = df.Sex.astype('int64')

    #embark c=0, s=1, q=2
    df.loc[df.Embarked == 'C', 'Embarked'] = 0
    df.loc[df.Embarked == 'S', 'Embarked'] = 1
    df.loc[df.Embarked == 'Q', 'Embarked'] = 2
    #All missing ebmbarks just make them embark from most common place
    df.Embarked = df.Embarked.fillna(np.round(df.Embarked.mean()))
    df.Embarked = df.Embarked.astype('int64')

    #I need to fill in the gaps of the data and make it complete.
    #So where there is no price, I will assume price on median of that class
    # Where there is no age I will give median of all ages
    df.Age = df.Age.fillna(df.Age.median())

    #All the missing prices assume median of their respectice class
    df.Fare = df.Fare.fillna(0)
    for i in xrange(1, df.Pclass.nunique()+1):
        mean_price = df.loc[(df.Fare != 0) & (df.Pclass == i), 'Fare'].mean()
        df.loc[(df.Fare == 0) & (df.Pclass == i), 'Fare'] = mean_price

    #remove the name data, cabin and ticket
    del df['Name']
    del df['Cabin']
    del df['Ticket']


#The data is now ready to go. So lets train then test!
train_data = train_data[cols]
test_data = test_data[cols]

# Analyzing
total_women = float(train_data.Sex.loc[train_data.Sex == 0].count())
total_men = float(train_data.Sex.loc[train_data.Sex == 1].count())
passengers = float(train_data.Sex.count())

surviving_women = float(train_data.loc[(train_data.Sex == 0) & (train_data.Survived == 1), 'Survived'].count())
surviving_men = float(train_data.loc[(train_data.Sex == 1) & (train_data.Survived == 1), 'Survived'].count())
total_survivors = float(surviving_women + surviving_men)

print 'Proportion of women who survived is %0.2f' % (surviving_women / total_women)
print 'Proportion of men who survived is %0.2f' % (surviving_men / total_men)
print 'Overall survived %0.2f' % (total_survivors / passengers)

prices = train_data.Fare.values.astype(np.float)
survival = train_data.Survived.values.astype(np.float)

logprices = np.log(prices)

# Prices Bins
bins = []
for i in range(0, numbins):
    bins.append(np.percentile(logprices, round(100.0*i/numbins)))

count = 0
survivalrate = []
midprice = []  # midprice (mid log price per bin)

for i in range(len(bins)-1, -1, -1):
    currentprices = (logprices > bins[i])
    survivalrate.append(np.sum(survival[currentprices]) / np.size(survival[currentprices]))
    midprice.append(np.median(logprices[currentprices]))
    #print i, ":", bins[i], ":", int(np.sum(survival[currentprices])), "/", int(np.size(survival[currentprices])), "=", \
    #  np.sum(survival[currentprices]) / np.size(survival[currentprices])
    # count = count + int(np.size(survivals[currentprices])) # DEBUG
    logprices[currentprices] = -1 # ignore - too low for lowest bin

# From list to array, and sort
survivalrate = np.array(survivalrate).astype(np.float)
survivalrate.sort()
midprice.sort()
binsarray = np.array(midprice).astype(np.float)

model_array = []
print(type(midprice))
for x in midprice:
    model_array.append([x])
model_array = np.array(model_array).astype(np.float)

clf = linear_model.LinearRegression()
clf.fit(model_array, survivalrate)
a = clf.coef_[0]
b = clf.intercept_
formula = "y = "+str(round(a,2))+"x + "+str(round(b,2))+"  (R^2 ="+str(round(clf.score(model_array, survivalrate),2))+")"
print formula
print "R^2 =", clf.score(model_array, survivalrate)

pl.scatter(midprice, survivalrate, s=5, label="tickets")
pl.xlabel("log price")
pl.ylabel("survival rate")
pl.legend(loc="best")
pl.title(formula)
pl.plot([1.8, 5], [1.8*a + b, a * 5 + b], c="b", label=formula)
pl.show()

print 'Training '
#forest = RandomForestClassifier(n_estimators=1000)

#valuestofeed = train_data.iloc[:, 2:].values
#resultstofeed = train_data.iloc[:, 0].values
#testvalues = test_data.iloc[:, 2:].values

#forest = forest.fit(valuestofeed, resultstofeed)

print 'Predicting'
#prediction = forest.predict(testvalues)

#test_data.Survived = prediction

#output = test_data.iloc[:, 0:2]
#output = output[['PassengerId', 'Survived']]
#output.to_csv('Result\\ForestPandas.csv', index=False)
