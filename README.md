# MLB First Pitch
# Overview
The purpose of this project is to create machine learning models for predicting the result of the first pitch of Major Leage Baseball (MLB) at-bats. The models will categorize the results of the first pitch into either "strike", "ball", or "hit into play" in multiclass models or "strike/ball" or "hit into play" for two-class models. The project utilizes MLB Statcast data accessed via the "pybaseball" python package. The Statcast data includes many features for the location and result of the pitch that could be used to improve performance of the model, but inclusion of these feature would make the models reflective rather than predictive. The goal is to develop a reliable tool that could predict first pitch results based off only features that could be known prior to the start of an at-bat, primarily focusing on game situation and pitcher tendencies. This could be used to set batter or fielder expectations or be employed in gambling markets that allow for the prediction of first pitch results. 
# Content
1. Two-Class Models <br>
   a. First_Pitch_Single_Batter.ipynb: Trains decision tree, random forest, artificial neural network, XGBoost, LightGBM, and CatBoost models on data for a single batter. Applies the models to a set of validation data withheld from the train/test portion of the code to prevent leakage. <br>
   b. First_Pitch_Multi_Batter.ipynb: Trains the individual models on each batter above thresholds of at-bats and first pitch hit into play percentage. Evaluates which model is most effective for the validation data. <br>
   c. First_Pitch_Global.ipynb: Trains the same models on a dataset of all batters above thresholds of at-bats and first pitch <br>
2. Multiclass Models <br>
   a. First_Pitch_Single_Batter_Multiclass.ipynb: <br>
   b. First_Pitch_Multi_Batter_Multiclass.ipynb: <br>
   c. First_Pitch_Global_Multiclass.ipynb: <br>
3. Python Scripts <br>
   a. statcast_loader.py: <br>
   b. build_context_features.py: <br>
   c. train_global_model.py: <br>
   d. predict_first_pitch_global.py: <br>
   e. evaluate_game.py: <br>
