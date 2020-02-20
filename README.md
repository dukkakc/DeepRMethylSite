# DeepRMethylSite

DeepRMethylSite : Prediction of Arginine Methylation in Proteins using Deep Learning. Devoloped in KC lab.
# Requirement
  Backend = Tensorflow <br/>
  Keras <br/>
  Numpy <br/>
  Biopython <br/>
  Sklearn <br/>
  Imblearn <br/>
 # Dataset
 Dataset is in FASTA format which includes protein window size of 51. Test dataset is provided. There are two dataset for positive and negative examples.
 # Model
 The best model for both CNN and LSTM have been included. Download all parts and extracting the 1st part will automatically extract remaining parts. The final files will be model_best_cnn.h5 and model_best_lstm.h5
 # Prediction for given test dataset
 With all the prerequisite installed, run -> model_gridsearch_load.py
 # Prediction for your dataset
 The format should be same as the test dataset which is basically FASTA format. This model works for window size 51 only. 
 # Contact 
 Feel free to contact us if you need any help : dukka.kc@wichita.edu
