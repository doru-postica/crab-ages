
# Crab age prediction project
***
Given 8 features of a crab, predict its age (1 to 29 months as per training data set).

This project is based on Kaggle challenge available [here](https://www.kaggle.com/competitions/playground-series-s3e16/overview).

###  📚 **Data preprocessing and model choice.**
* Feature engineering: created 12 feature interactions and *2, *3, *1/2 polynomials
* Used Local Outlier Factor to check outliers
* Used StandardScaler + KNN Imputer for numerical features
* Used SimpleImputer + OHE for categorical features
* Hypertuned an XGBRegressor model using a Pipeline within 5-fold GridSearchCV

*Working file and training data can be found in "eda" folder.*

### ⭐️ Performance
* Kaggle metric was MAE
  * Training data: 1.374
  * Test data: 1.381
  * Kaggle personal submission: 1.372
  * Kaggle top result: 1.334
  * Leaderboard: 628/1429

### 🚀 Model deployment
* Deployed on Streamlit, live version can be found [here](https://crab-ages-prediction.streamlit.app/).
* User input is done via sliders.
  * If any slider value is 0, a KNNImputer (fit on our training data) is used.
* Added confidence levels (chosen by user via slider) to our prediction via MAPIE library.
* Before predicting, we assess similarity of user input to our training data.
  * We compute average distance of user input to closest 5 points of training data.
  * We compare this distance to those computed for our points.
  * If distance is 1.5 x IQR above the 75th percentile, we display a warning message.
  * If distance is 3 x IQR above the 75th percentile, we do not display prediction at all.