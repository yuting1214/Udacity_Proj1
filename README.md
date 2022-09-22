# Predict Bike Sharing Demand with AutoGluon 

## Initial Training

The initial training consists of three parts.

1.  Preprocessing: Parse the dataime variable into datetime64
2.  Training: Execute AutoGluon
    * Automatical feature engineering to decompose dataime into ['datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    * Iterate a set of models
3. Predicting: Pick the best model based on evaluation score to predcit.

**Make sure values of prediction in the submission file all larger than zero.**

Summary of trainning results:

|    | model               | score_val   | pred_time_val | fit_time  | pred_time_val_marginal | fit_time_marginal | stack_level | can_infer | fit_order |   |
|----|---------------------|-------------|---------------|-----------|------------------------|-------------------|-------------|-----------|-----------|---|
| 0  | WeightedEnsemble_L2 | -92.389673  | 0.290135      | 0.318677  | 0.000000               | 0.284673          | 2           | True      | 12        |   |
| 1  | KNeighborsDist      | -92.389673  | 0.290135      | 0.034004  | 0.290135               | 0.034004          | 1           | True      | 2         |   |
| 2  | KNeighborsUnif      | -109.626075 | 0.128549      | 0.044039  | 0.128549               | 0.044039          | 1           | True      | 1         |   |
| 3  | RandomForestMSE     | -121.961896 | 0.111347      | 2.304636  | 0.111347               | 2.304636          | 1           | True      | 5         |   |
| 4  | ExtraTreesMSE       | -128.646118 | 0.112063      | 0.950567  | 0.112063               | 0.950567          | 1           | True      | 7         |   |
| 5  | LightGBMLarge       | -132.173561 | 0.007729      | 1.432540  | 0.007729               | 1.432540          | 1           | True      | 11        |   |
| 6  | LightGBM            | -134.080427 | 0.020078      | 1.020219  | 0.020078               | 1.020219          | 1           | True      | 4         |   |
| 7  | CatBoost            | -134.236163 | 0.006387      | 10.707531 | 0.006387               | 10.707531         | 1           | True      | 6         |   |
| 8  | XGBoost             | -135.075087 | 0.008754      | 1.924407  | 0.008754               | 1.924407          | 1           | True      | 9         |   |
| 9  | LightGBMXT          | -135.958034 | 0.021801      | 5.584855  | 0.021801               | 5.584855          | 1           | True      | 3         |   |
| 10 | NeuralNetFastAI     | -136.531080 | 0.030514      | 18.338974 | 0.030514               | 18.338974         | 1           | True      | 8         |   |
| 11 | NeuralNetTorch      | -140.125794 | 0.028239      | 74.514935 | 0.028239               | 74.514935         | 1           | True      | 10        |   |

### What was the top ranked model that performed?

According to the above table, the top ranked model is WeightedEnsemble_L2, the two layers ensemble model.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
TODO: Add your explanation

### How much better did your model preform after adding additional features and why do you think that is?
TODO: Add your explanation

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
TODO: Add your explanation

### If you were given more time with this dataset, where do you think you would spend more time?
TODO: Add your explanation

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|?|?|?|?|
|add_features|?|?|?|?|
|hpo|?|?|?|?|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](img/model_test_score.png)

## Summary
TODO: Add your explanation
