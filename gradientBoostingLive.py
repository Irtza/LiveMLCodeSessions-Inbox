import numpy as np
from sklearn import datasets

import xgboost

from sklearn.ensemble import GradientBoostingClassifier
datasets.load_iris()
dataset = datasets.load_iris()
gbc = GradientBoostingClassifier
xgb = xgboost
gbc = GradientBoostingClassifier?
xgb = xgboost?
xgb = xgboost()
xgb = xgboost.XGBClassifier?
xgb = xgboost.XGBClassifier()
xgb = xgb.fit(dataset.data)
xgb.fit?
xgb.fit(dataset.data , dataset.target)
xgb = xgb.fit(dataset.data , dataset.target)
xgb = xgb.fit(dataset.data[:100,:] , dataset.target[:,100])
xgb = xgb.fit(dataset.data[:100,:] , dataset.target[:100])
dataset.target
xgb = xgb.fit(dataset.data,dataset.target)
pred = xgb.predict(dataset.data)
pred
xgb.feature_importances_
xgb.score?
xgb.score(dataset.data,dataset.target)
imp = xgb.feature_importances_
from matplotlib import pyplot as plt
plt.plot(imp)
plt.show()
dataset.feature_names
dataset.feature_names[np.argmax(xgb.feature_importances_)]
