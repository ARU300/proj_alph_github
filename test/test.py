from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df)
import revoscalepy
from revoscalepy import rx_summary
summary = rx_summary("petal length (cm)", df)
print(summary)
# Import the DeployClient and MLServer classes from 
# the azureml-model-management-sdk package so you can 
# connect to Machine Learning Server (use=MLServer).

from azureml.deploy import DeployClient
from azureml.deploy.server import MLServer

# Define the location of the Machine Learning Server
HOST = 'http://localhost:12800'
# And provide your username and password as a Python tuple
# for local admin with password Pass123!
# context = ('admin', 'Pass123!')
context = ('admin', 'Abinaya25@')
client = DeployClient(HOST, use=MLServer, auth=context)