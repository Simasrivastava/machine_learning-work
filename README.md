# machine_learning-work
df=pd.read_csv('House_price.csv')
df
from sklearn.datasets import fetch_20newsgroups
data=fetch_20newsgroups()
print(data.target_names)
print('\n')
categories=['talk.religion.misc','soc.religion.christian','sci.space','comp.graphics']
train=fetch_20newsgroups(subset='train',categories=categories)
test=fetch_20newsgroups(subset='test',categories=categories)
print(train.data[5])
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
model=make_pipeline(TfidVectorizer(),MultinomialNB())
model.fit(train.data,train.target)
labels=model.predict(test.data)
print('==================')
print(accuracy_score(test.target,labels))
from sklearn.metrics import confusion_matrix
mat=confusion_matrix(test.target,labels)
sns.heatmap(mat.T,square=True,annot=True,Fmt='d',cbar=False,xticklabels=train.target_names,yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicated label')
def predict_category(s,train=train,model=model):
    pred=model.predict([s])
    print("*******************",pred)
    return train.target_names[pred[0]]
print('--------------category----------------')
print(predict_category('sending a payload to the ISS'))
print(predict_category('discussing islam'))
df=pd.read_csv('House_price.csv')
x=df['bedrooms']
y=df['price']
X=x[:,np.newaxis]
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=1)
model=LinearRegression(fit_intercept=True)
model.fit(Xtrain,ytrain)
y_model=model.predict(Xtest)
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print('R Square Error=',r2_score(ytest,y_model))
print('\n MAE=',mean_absolute_error(ytest,y_model))
print('\n Root Mean Square Error=',np.sqrt(mean_squared_error(ytest,y_model)))
df=pd.read_csv('House_price.csv')
x=df['bedrooms']
y=df['price']
X=x[:,np.newaxis]
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=1)
model=LinearRegression(fit_intercept=True)
model.fit(Xtrain,ytrain)
y_model=model.predict(Xtest)
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print('R Square Error=',r2_score(ytest,y_model))
print('\n MAE=',mean_absolute_error(ytest,y_model))
print('\n Root Mean Square Error=',np.sqrt(mean_squared_error(ytest,y_model)))
df.corr()
df=pd.read_csv("training.csv")
df
###polynomial regression 
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
df=pd.read_csv('House_price.csv')
x=df['bedrooms']
y=df['price']
X=x[:,np.newaxis]
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=1)
poly=PolynomialFeatures(degree=5)
Xtrain_poly=poly.fit_transform(Xtrain)
Xtest_poly=poly.fit_transform(Xtest)
#create regression using linear regression
regressor=LinearRegression()
#training our model
regressor.fit(Xtrain_poly,ytrain)
#predicting values using our trained model
y_pred=regressor.predict(Xtest_poly)
r_2_score=r2_score(ytest,y_pred)
print("R Squared Score"+str(r_2_score))
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
dataset=datasets.load_iris()
model=GaussianNB()
x=dataset.data
y=dataset.target
model.fit(x,y)
y_pred=model.predict(x)
print(metrics.classification_report(y,y_pred))
print(metrics.confusion_matrix(y,y_pred))
mat=metrics.confusion_matrix(y,y_pred)
from sklearn.metrics import accuracy_score
print('\n Accuracy Score is=',accuracy_score(dataset.target,y_pred))
sns.heatmap(mat,square=True,annot=True,fmt='d',cbar=False,xticklabels=dataset.target_names,yticklabels=dataset.target_names)
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
dataset=datasets.load_iris()
model=BernoulliNB()
x=dataset.data
y=dataset.target
model.fit(x,y)
y_pred=model.predict(x)
print(metrics.classification_report(y,y_pred))
print(metrics.confusion_matrix(y,y_pred))
mat=metrics.confusion_matrix(y,y_pred)
from sklearn.metrics import accuracy_score
print('\n Accuracy Score is=',accuracy_score(dataset.target,y_pred))
sns.heatmap(mat,square=True,annot=True,fmt='d',cbar=False,xticklabels=dataset.target_names,yticklabels=dataset.target_names)
from sklearn.datasets import fetch_20newsgroups
data=fetch_20newsgroups()
print(data.target_names)
print('\n')
categories=['talk.religion.misc','soc.religion.christian','sci.space','comp.graphics']
train=fetch_20newsgroups(subset='train',categories=categories)
test=fetch_20newsgroups(subset='test',categories=categories)
print(train.data[5])
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
model=make_pipeline(Tfidvectorizer(),MultinomialNB())
model.fit(train.data,train.target)
labels=model.predict(test.data)
print('==================')
print(accuracy_score(test.target,labels))
from sklearn.metrics import confusion_matrix
mat=confusion_matrix(test.target,labels)
sns.heatmap(mat.T,square=True,annot=True,Fmt='d',cbar=False,xticklabels=train.target_names,yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicated label')
def predict_category(s,train=train,model=model):
    pred=model.predict([s])
    print("*******************",pred)
    return train.target_names[pred[0]]
print('--------------category----------------')
print(predict_category('sending a payload to the ISS'))
print(predict_category('discussing islam'))
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
iris=datasets.load_iris()
x=iris.data[:,[2,3]]
y=iris.target
sc=StandardScaler()
x=sc.fit_transform(x)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)
model=SVC(kernel='linear',C=1.0,random_state=0).fit(xtrain,ytrain)
print('Train Accuracy score',metrics.accuracy_score(ytrain,model.predict(xtrain)))
print('Train Confusion metrix\n',metrics.confusion_matrix(ytrain,model.predict(xtrain)))
print('\n')
print('Train Accuracy score',metrics.accuracy_score(ytest,model.predict(xtest)))
print('Train Confusion metrix\n',metrics.confusion_matrix(ytest,model.predict(xtest)))
print('Classification Report\n',metrics.classification_report(ytest,model.predict(xtest)))
