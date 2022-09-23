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

### Work on Features

#### Transform existing features

The features of "season" and "weather" are actually categorial but recongnized initially as numerical. Therefore, transform these two features to help model better utilize them when training.

#### Create new features

![hourly count](https://github.com/yuting1214/Udacity_Proj1/blob/main/plots/hour_dist.png)

There is an obvious pattern in the hourly count even in different year. Therefore, to increase the complexity of the models, we create a new categorial feature called "hour" from datatime to differentiate the demand count in different hour.

### How much better did your model preform after adding additional features and why do you think that is?
| model | score_val                         | pred_time_val | fit_time  | pred_time_val_marginal | fit_time_marginal | stack_level | can_infer | fit_order | fit_order |   |
|-------|-----------------------------------|---------------|-----------|------------------------|-------------------|-------------|-----------|-----------|-----------|---|
| 0     | WeightedEnsemble_L3               | -28.405767    | 16.580596 | 906.125983             | 0.001054          | 0.844273    | 3         | True      | 19        |   |
| 1     | NeuralNetFastAI_BAG_L2            | -28.479117    | 15.927134 | 666.571042             | 0.884139          | 42.337493   | 2         | True      | 18        |   |
| 2     | LightGBMXT_BAG_L2                 | -30.496543    | 16.635251 | 634.827226             | 1.592256          | 10.593677   | 2         | True      | 13        |   |
| 3     | CatBoost_BAG_L2                   | -30.612008    | 15.259786 | 854.975517             | 0.216791          | 230.741968  | 2         | True      | 16        |   |
| 4     | LightGBM_BAG_L2                   | -30.644736    | 15.478612 | 632.202249             | 0.435617          | 7.968700    | 2         | True      | 14        |   |
| 5     | ExtraTreesMSE_BAG_L2              | -32.637065    | 16.114563 | 627.123394             | 1.071568          | 2.889845    | 2         | True      | 17        |   |
| 6     | RandomForestMSE_BAG_L2            | -32.785444    | 15.991667 | 631.278578             | 0.948672          | 7.045029    | 2         | True      | 15        |   |
| 7     | WeightedEnsemble_L2               | -33.437004    | 12.522007 | 563.113157             | 0.001997          | 0.949021    | 2         | True      | 12        |   |
| 8     | LightGBM_BAG_L1                   | -35.724454    | 3.020893  | 9.857582               | 3.020893          | 9.857582    | 1         | True      | 4         |   |
| 9     | LightGBMXT_BAG_L1                 | -36.181212    | 7.431891  | 13.941288              | 7.431891          | 13.941288   | 1         | True      | 3         |   |
| 10    | CatBoost_BAG_L1                   | -36.268528    | 0.291619  | 492.539192             | 0.291619          | 492.539192  | 1         | True      | 6         |   |
| 11    | NeuralNetFastAI_BAG_L1            | -36.474018    | 0.695772  | 41.922104              | 0.695772          | 41.922104   | 1         | True      | 8         |   |
| 12    | ExtraTreesMSE_BAG_L1              | -38.363770    | 0.918801  | 2.239453               | 0.918801          | 2.239453    | 1         | True      | 7         |   |
| 13    | RandomForestMSE_BAG_L1            | -38.438708    | 0.959705  | 3.852583               | 0.959705          | 3.852583    | 1         | True      | 5         |   |
| 14    | XGBoost_BAG_L1                    | -40.322325    | 0.771656  | 18.820589              | 0.771656          | 18.820589   | 1         | True      | 9         |   |
| 15    | NeuralNetTorch_BAG_L1             | -44.783546    | 0.640904  | 36.390091              | 0.640904          | 36.390091   | 1         | True      | 10        |   |
| 16    | KNeighborsDist_BAG_L1             | -84.146423    | 0.120130  | 0.051387               | 0.120130          | 0.051387    | 1         | True      | 2         |   |
| 17    | KNeighborsUnif_BAG_L1 -101.588176 | 0.123894      | 0.064907  | 0.123894               | 0.064907          | 1           | True      | 1         |           |   |
| 18    | LightGBMLarge_BAG_L1 -176.376584  | 0.067731      | 4.554373  | 0.067731               | 4.554373          | 1           | True      | 11        |           |   |

### Work on Response

Original response is skewed, so apply log transform to help better detect the relationship with features.

![Transformation](https://github.com/yuting1214/Udacity_Proj1/blob/main/plots/y_transform.png)

### Train with scikit-learn
```
Model: Random Forest Regressor with 200 trees.
Preprocessing:
   New feature: "hour"
   One-hot encoding:  ['season', 'holiday', 'weather', 'year', 'month', 'hour']
Hyperparameter:
   n_estimators(trees): 200
Evaluation:
   RMSLE: 0.42584(leaderboard)
```

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
```
Use LightGBM to conduct Hyper parameter tuning

gbm_options = {  # specifies non-default hyperparameter values for lightGBM gradient boosted trees
    'num_boost_round': 300,  # number of boosting rounds (controls training time of GBM models) default:100
    'num_leaf': ag.space.Int(lower=20, upper=66, default=31),  # number of leaves in trees (integer hyperparameter)
    'reg_alpha': ag.space.Real(0.0, 0.1, default=0.0)

}
```

Submit the prediction from best model: 0.66786(publicScore).

So the model doesn't do well compare to simple Random Forest model.


### If you were given more time with this dataset, where do you think you would spend more time?

1. Add a feature related to whether rain or not.
2. Check specfic holidays in the datetime that are not recoreded
3. Try [MARS](https://contrib.scikit-learn.org/py-earth/content.html#multivariate-adaptive-regression-splines)

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.

| fileName                         | date                | description                       | status   | publicScore | privateScore | 
|----------------------------------|---------------------|-----------------------------------|----------|-------------|--------------|
| submission_new_hpo.csv           | 2022-09-23 02:15:44 | new features with hyperparameters | complete | 0.66786     | 0.66786      | 
| submission_new_features_logy.csv | 2022-09-22 17:20:53 | new features log y                | complete | 0.96457     | 0.96457      |  
| submission_lag1_log.csv          | 2022-09-22 16:22:58 | lag1 log y                        | complete | 0.42835     | 0.42835      |  
| submission_rf_log.csv            | 2022-09-22 05:01:29 | random forest log y               | complete | 0.42584     | 0.42584      |  
| submission_rf.csv                | 2022-09-22 04:55:06 | random forest                     | complete | 0.54418     | 0.54418      |  
| submission_rmsle_new.csv         | 2022-09-22 00:19:58 | rmsle_update                      | complete | 0.72473     | 0.72473      |
| submission_rmsle.csv             | 2022-09-21 23:49:51 | rmsle                             | complete | 0.70231     | 0.70231      |  
| submission_new_features.csv      | 2022-09-21 22:31:06 | new features                      | complete | 0.54823     | 0.54823      |  
| submission.csv                   | 2022-09-21 21:16:35 | first raw submission              | complete | 1.84672     | 1.84672      |  

| Brief | Detail | 
|-------------|--------------|
| first raw submission | Parse date, use AutoGluon directly |
| new features | Categorize: season, weather; Add hour |
| rmsle | Replace eval_metric with self-defined rmsle in AutoGluon  |
| rmsle_update  | Replace eval_metric with self-defined rmsle in AutoGluon and make sure use new features  |
| random forest | Use RF in scikit-learn and features as "new features" |
| random forest log y | Same as "random forest" but transform y with log |
| lag1 log y | Use "random forest" to create a lag one feature for test data and train a "random forest" with lag one count feature |
| new features log y | Same as "new features"  but transform y with log |
| new features with hyperparameters | Hyperparameter tunning on lightGBM |

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](https://github.com/yuting1214/Udacity_Proj1/blob/main/plots/train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![model_test_score.png](https://github.com/yuting1214/Udacity_Proj1/blob/main/plots/test_score.png)

## Summary

Autogluon is a convenient tool when attempting to execute bunch of powerful models simultanouely.
Howeever, its flexibility brings some complexity when. Therefore, it would better start from some go-to methods like RandomForest first. And use Autoluoon as auxiliary tool when trying to increate the complexitiy of the model.
