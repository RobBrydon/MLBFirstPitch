# MLB First Pitch
# Purpose
The purpose of this project is to create machine learning models for predicting the result of the first pitch of MLB at-bats. The models will categorize the results of the first pitch into either "strike", "ball", or "hit into play" in multiclass models or "strike/ball" or "hit into play" for two-class models.
# Content
1. Two-Class Models
   a. First_Pitch_Single_Batter.ipynb: Trains decision tree, random forest, artificial neural network, XGBoost, LightGBM, and CatBoost models on data for a single batter. Applies the models to a set of validation data withheld from the train/test portion of the code to prevent leakage.
   b. First_Pitch_Multi_Batter.ipynb: Trains the individual models on each batter above thresholds of at-bats and first pitch hit into play percentage. Evaluates which model is most effective for the validation data.
   c. First_Pitch_Global.ipynb: Trains the same models on a dataset of all batters above thresholds of at-bats and first pitch.
2. Multiclass Models
   a. First_Pitch_Single_Batter_Multiclass.ipynb:
   b. First_Pitch_Multi_Batter_Multiclass.ipynb:
   c. First_Pitch_Global_Multiclass.ipynb:
3. Python Scripts
   a. statcast_loader.py: 
   b. build_context_features.py: 
   c. train_global_model.py: 
   d. predict_first_pitch_global.py: 
   e. evaluate_game.py: 
