from numpy import *
import operator
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import *
url = "iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
# Only the X variables
data = df[['sepal length','sepal width','petal length','petal width']]
#calculate SVD
n = 2 # We will take two Singular Values
U, s, V = linalg.svd( data ) #Singular Value Decomposition
# eye() creates a matrix with ones on the diagonal and zeros elsewhere
Sig = mat(eye(n)*s[:n])
newdata = U[:,:n]
newdata = pd.DataFrame(newdata)
newdata.columns=['SVD1','SVD2']
newdata.head()

# Add the actual target to the data in order to plot it
newdata['target']=df['target']
fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('SVD 1') 
ax.set_ylabel('SVD 2') 
ax.set_title('SVD') 
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = newdata['target'] == target
    ax.scatter(newdata.loc[indicesToKeep, 'SVD1'], newdata.loc[indicesToKeep, 'SVD2'], c = color, s = 50)
ax.legend(targets)
ax.grid()