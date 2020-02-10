# Code written by Jerin Rajan on 01st Feb 2019.
# Build a model
# Inputs (21 attributes)
# Output - y (outcome of the client)

import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize

# INITITIALISATION OF VARIABLES
i = 0
x = dict()
key = list()
value = list()
var_x = list()


# DATASET
# ====================================================
# Filename of the datasource CSV file
fname = 'data set.csv'
# load the dataset file
data = pd.read_csv(fname)

# Dataset Information
print('Dataset Info:\n',data.info())
# Print top 100 rows of data
print(data.head(100))

# Shows the header items
print('Data columns:\n',data.columns)

# # Descriptions - Mean, Std, Min, Max, Percentiles
# print('Data Descriptions:\n',data.describe())


# FEATURE ENGINEERING - Convert variable categories to numeric
# ====================================================
# Loading the original dataset to a new variable
data_new = data

# Numerical conversion
data_new.job = pd.Categorical(data_new.job)
data_new.marital = pd.Categorical(data_new.marital)
data_new.education = pd.Categorical(data_new.education)
data_new.default = pd.Categorical(data_new.default)
data_new.housing = pd.Categorical(data_new.housing)
data_new.loan = pd.Categorical(data_new.loan)
data_new.contact = pd.Categorical(data_new.contact)
data_new.month = pd.Categorical(data_new.month)
data_new.day_of_week = pd.Categorical(data_new.day_of_week)
data_new.poutcome = pd.Categorical(data_new.poutcome)
data_new.y = pd.Categorical(data_new.y)

data_new['job'] = data_new.job.cat.codes
data_new['marital'] = data_new.marital.cat.codes
data_new['education'] = data_new.education.cat.codes
data_new['default'] = data_new.default.cat.codes
data_new['housing'] = data_new.housing.cat.codes
data_new['loan'] = data_new.loan.cat.codes
data_new['contact'] = data_new.contact.cat.codes
data_new['month'] = data_new.month.cat.codes
data_new['day_of_week'] = data_new.day_of_week.cat.codes
data_new['poutcome'] = data_new.poutcome.cat.codes
data_new['y'] = data_new.y.cat.codes

# Display the numeric converted dataset
print('Numeric converted Datasets:\n',data_new.head(20))

# Descriptions - Mean, Std, Min, Max, Percentiles
print('Data Descriptions:\n',data.describe())

# UNIVARIATE PLOTS
# ====================================================
# Distribution of each individual variables
# 1) BOXPLOT
data_new.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
# Plotting the boxplot figure
plt.figure(num=1,figsize=(15.0,10.0)).tight_layout(pad=0.1,w_pad=0.4)
# Title of the Boxplot
plt.title('Box Plot',pad=1.0)
# Save the boxplot figure
plt.savefig('BoxPlot.png')

# 2) HISTOGRAM
data_new.hist()
# Plot the histogram figure
plt.figure(num=2,figsize=(15.0,10.0)).tight_layout(pad=0.1,w_pad=0.4)
# Title of the histogram
plt.title('Histogram Plot',pad=1.0)
# Saving the histogram figure
plt.savefig('HistogramPlot.png')

# MULTIVARIATE PLOTS - Identify any highly correlated pairs of variables
# ====================================================
# SCATTER PLOT
# 1) Bank client Data
data_customer = {'age':data_new['age'],'job':data_new['job'],
                'marital':data_new['marital'],'education':data_new['education'],
                'default':data_new['default'], 'housing':data_new['housing'],
                'loan':data_new['loan']}

# converting to dataset format
df_customer = pd.DataFrame(data=data_customer)

# DATA TRANSFORMATION
# Scaling the data
min_max_scaler = MinMaxScaler()
# Scaled data
df_customer_scaled = min_max_scaler.fit_transform(df_customer)
# converting to dataset format
df_customer_scaled_dt = pd.DataFrame(data=df_customer_scaled,
                columns=['age','job','marital','education',
                'default','housing','loan'])
# Plotting Scaled data within a scatter plot matrix
fig = pd.plotting.scatter_matrix(df_customer_scaled_dt)
# Title of the scatter plot matrix
plt.suptitle('Scatter Plot - Client Data')
# Dimensions of the figure and fitting the subplots
plt.figure(num=3,figsize=(15.0,10.0)).tight_layout(pad=0.1,w_pad=0.4)
# Saving the scatter plot
plt.savefig('Scatter Plot - Client Data.png')


# 2) Variables related to the last contact of the current campaign
data_current_campaign = {'contact':data_new['contact'],'month':data_new['month'],
            'day_of_week':data_new['day_of_week'],'duration':data_new['duration']}
# converting to dataset format
df_current_campaign = pd.DataFrame(data=data_current_campaign)
# Scaling the data
df_current_campaign_scaled = min_max_scaler.fit_transform(df_current_campaign)
# Scaled data
df_current_campaign_scaled_dt = pd.DataFrame(data=df_current_campaign_scaled,
                columns=['contact','month','day_of_week','duration'])
# Plotting Scaled data within a scatter plot matrix
fig = pd.plotting.scatter_matrix(df_current_campaign_scaled_dt)
# Title of the scatter plot matrix
plt.suptitle('Scatter Plot - Campaign Data')
# Dimensions of the figure and fitting the subplots
plt.figure(num=4,figsize=(15.0,10.0)).tight_layout(pad=0.1,w_pad=0.4)
# Saving the scatter plot
plt.savefig('Scatter Plot - Campaign Data.png')

# 3) Other attributes related to the campaign(s)
data_camp_other_attr = {'campaign':data_new['campaign'],'pdays':data_new['pdays'],
            'previous':data_new['previous'],'poutcome':data_new['poutcome']}
# converting to dataset format
df_camp_other_attr = pd.DataFrame(data=data_camp_other_attr)
# Scaling the data
df_camp_other_attr_scaled = min_max_scaler.fit_transform(df_camp_other_attr)
# Scaled data
df_camp_other_attr_scaled_dt = pd.DataFrame(data=df_camp_other_attr_scaled,
                columns=['campaign','pdays','previous','poutcome'])
# Plotting Scaled data within a scatter plot matrix
fig = pd.plotting.scatter_matrix(df_camp_other_attr_scaled_dt)
# Title of the scatter plot matrix
plt.suptitle('Scatter Plot - Campaign Other attributes')
# Dimensions of the figure and fitting the subplots
plt.figure(num=5,figsize=(15.0,10.0)).tight_layout(pad=0.1,w_pad=0.4)
# Saving the scatter plot
plt.savefig('Scatter Plot - Campaign Other attributes.png')

# 4) Social and economic context attributes
data_soc_eco_attr = {'emp.var.rate':data_new['emp.var.rate'],
            'cons.price.idx':data_new['cons.price.idx'],
            'cons.conf.idx':data_new['cons.conf.idx'],
            'euribor3m':data_new['euribor3m'],
            'nr.employed':data_new['nr.employed']}
# converting to dataset format
df_soc_eco_attr = pd.DataFrame(data=data_soc_eco_attr)
# Scaling the data
df_soc_eco_attr_scaled = min_max_scaler.fit_transform(df_soc_eco_attr)
# Scaled data
df_soc_eco_attr_scaled_dt = pd.DataFrame(data=df_soc_eco_attr_scaled,
            columns=['emp.var.rate','cons.price.idx',
            'cons.conf.idx','euribor3m','nr.employed'])
# Plotting Scaled data within a scatter plot matrix
fig = pd.plotting.scatter_matrix(df_soc_eco_attr_scaled_dt)
# Title of the scatter plot matrix
plt.suptitle('Scatter Plot - Social & Economic attributes')
# Dimensions of the figure and fitting the subplots
plt.figure(num=6,figsize=(15.0,10.0)).tight_layout(pad=0.1,w_pad=0.4)
# Saving the scatter plot
plt.savefig('Scatter Plot - Social & Economic attributes.png')

# MULTIVARIATE PLOTS - CORRELATION MATRIX
# Identify which variable is the most predictive of a client’s subscription
# Correlation matrix data
corr_data_dt = data_new.corr()
# Correlation matrix data
print('correlation_matrix:\n',corr_data_dt )
# Correlation of 'y' data
print('Correlation of y:\n',corr_data_dt['y'])
# # Sorting the correlation of y in ascending order
# sort_y = np.sort(corr_data_dt['y'])
# print('asc_sort:\n',sort_y)

# Dictionary of the y correlation matrix with the associated x variables
for i in range(len(corr_data_dt['y'])):
    key = corr_data_dt['y'][i]
    value = data_new.columns[i]
    x[key] = value

# Correlation y with x variables
sorted = np.sort(corr_data_dt['y'])
# Correlation in descending order
reverse_array = sorted [::-1]
print('Desc order:\n',reverse_array)
# Top 5 Correlated values with Y in descending order
top_5 = reverse_array[1:6]
print('Most correlated with y:\n',top_5)

#Printing the most correlated x-variables to the output y
for i in range (len(top_5)):
    var_x.append(x[top_5[i]])
    print(x[top_5[i]])
# Adding the Y variable into the list
var_x.append('y')
# Plotting the Correlation Matrix - Figure.
f = plt.figure(num=7,figsize=(10, 8))
plt.matshow(corr_data_dt, fignum=f.number)
plt.xticks(range(data_new.shape[1]), data_new.columns, fontsize=7, rotation=45)
plt.yticks(range(data_new.shape[1]), data_new.columns, fontsize=7)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16, pad=30);
plt.savefig('Correlation Matrix.png')

# DATA PREPERATION
# ====================================================
# Split the input & output
dt = np.dtype('f')
data_array = np.array(data_new, dtype=dt)
X, y = data_array[:,:-1], data_array[:,-1]
# print('Input:',X.shape)
# print('Output:',y.shape)


# Training Sets - Split dataset
# Split dataset Training - 90%
# Split dataset Testing - 10%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1,random_state=0)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)


# Apply Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODEL -1: Logistic Regression
# ====================================================
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
# Measuring the Accuracy of the model using training set
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
# Measuring the Accuracy of the model using Test set
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))


# Model - 2: Logistic Regression Using 5 input variables only
# ====================================================
# 5 input variables are: day_of_week, duration, pdays, loan, poutcome
dt_new = data_new[var_x]
data_array2 = np.array(dt_new, dtype=dt)
X_2, y_2 = data_array2[:,:-1], data_array2[:,-1]
# print('Input:',X_2.shape)
# print('Output:',y_2.shape)

# Training Sets - Split dataset - 2
# Split dataset Training - 90%
# Split dataset Testing - 10%
X2_train, X2_test, y2_train, y2_test = train_test_split(X_2,y_2, test_size=0.1,random_state=0)
# print(X2_train.shape, y2_train.shape)
# print(X2_test.shape, y2_test.shape)

# Apply Scaling - 2
scaler = MinMaxScaler()
X2_train = scaler.fit_transform(X2_train)
X2_test = scaler.transform(X2_test)

# Build Model -2: Logistic Regression
logreg = LogisticRegression()
logreg.fit(X2_train, y2_train)
# Measuring the Accuracy of the model using training set
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X2_train, y2_train)))
# Measuring the Accuracy of the model using Test set
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X2_test, y2_test)))

# Show all plots on screen
plt.show()
