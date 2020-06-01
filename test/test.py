from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df)
#import revoscalpey
from revoscalepy import rx_summary
summary = rx_summary("petal length (cm)", df)
print(summary)