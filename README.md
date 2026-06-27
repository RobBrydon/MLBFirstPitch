# MLB First Pitch
# Overview
The purpose of this project is to create machine learning models for predicting the result of the first pitch of Major Leage Baseball (MLB) at-bats. The models will categorize the results of the first pitch into either "strike", "ball", or "hit into play" in multiclass models or "strike/ball" or "hit into play" for two-class models. The project utilizes MLB Statcast data accessed via the "pybaseball" python package. The Statcast data includes many features for the location and result of the pitch that could be used to improve performance of the model, but inclusion of these feature would make the models reflective rather than predictive. The goal is to develop a reliable tool that could predict first pitch results based off only features that could be known prior to the start of an at-bat, primarily focusing on game situation and pitcher tendencies. This could be used to set batter or fielder expectations or be employed in gambling markets that allow for the prediction of first pitch results.
# Content
1. Two-Class Models <br>
   a. First_Pitch_Single_Batter.ipynb: Trains decision tree, random forest, artificial neural network, XGBoost, LightGBM, and CatBoost models on data for a single batter. Applies the models to a set of validation data withheld from the train/test portion of the code to prevent leakage. This portion was primarily used to apply new features. <br>
   b. First_Pitch_Multi_Batter.ipynb: Trains the individual models on each batter above thresholds of at-bats (default: 100) and first pitch hit into play percentage (default: top 300). This file can be used to evaluate which model is most effective for the validation data. <br>
   c. First_Pitch_Global.ipynb: Trains the same models on a combined dataset of all batters above thresholds of at-bats (default: 100) and first pitch hit into play percentage (default: top 300). The global model provided the best results for predicting first pitch results. <br>
2. Multiclass Models <br>
   a. First_Pitch_Single_Batter_Multiclass.ipynb: Analogous to the two-class single-batter file for the multiclass models. <br>
   b. First_Pitch_Multi_Batter_Multiclass.ipynb: Analogous to the two-class multi-batter file for the multiclass models. <br>
   c. First_Pitch_Global_Multiclass.ipynb: Analogous to the two-class global model for the multiclass models. <br>
3. Python Scripts <br>
   a. statcast_loader.py: Caches Statcast data in a parquet file for specific date ranges to prevent repeated downloading from pybaseball. <br>
   b. build_context_features.py: Builds four context features (previous game pitcher pitch count, previous game batter first pitch hit into play at-bats count, current game pitcher pitch count, and current game batter first pitch hit into play count).  <br>
   c. train_global_model.py: Trains a global model for reference by other files.<br>
   d. predict_first_pitch_global.py: Uses the trained global model to predict the result of a new first pitch with user submitted variables.<br>
   e. evaluate_game.py: Evaluates the trained global model on all first pitches in a certain game.<br>
# Usage
For each of the single- and multi-batter models, the training and validation date ranges must be specified as a global variable. In addition, a decay rate must be specified with the default value set to 0.04, which sets data from 6 months earlier as half weight. In the single-batter models, the batter's name is also set as a global variable. Following loading and caching of the Statcast data, build_context_features.py should be run locally on the cached parquet file (example: "python build_context_features.py statcast_YYYYMMDD_YYYYMMDD.parquet"). Prior to running predict_first_pitch_global.py and evaluate_game.py, the train_global_model.py must be ran with a specified date range in the file. The evaluate_game.py file needs to have a date and teams specified in the file as well.
# Results

# Author
Robert Brydon, for inquiries or suggestions please contact via LinkedIn: <a href="https://www.linkedin.com/in/robert-brydon-phd-0241b5186/">Visit Robert's LinkedIn</a>
