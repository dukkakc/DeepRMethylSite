# DeepRMethylSite

DeepRMethylSite : Prediction of Arginine Methylation in Proteins using Deep Learning. Devoloped in KC lab.
# Requirement
  >=Python3.6
  Backend = Tensorflow <br/>
  Keras <br/>
  Numpy <br/>
  Biopython <br/>
  Sklearn <br/>
  Imblearn <br/>
  winrar or any compression program to open .rar file
 # Dataset
 Dataset are in fasta format. Both training and testing datasets are provided which are independent (one does not include others).
 Training dataset for positive and negative are train_s33_Pos_51.fasta and train_s33_Neg_51.fasta respectively. Testing dataset for positive and negative are test_s33_Pos_51.fasta and test_s33_Neg_51.fasta respectively. Training dataset is made available so that future models can train with it for the comparison.
 # Model
 The best trained model for both CNN and LSTM (used in our final results) have been included, model_best_cnn.h5 and model_best_lstm.h5 respectively. 
 # Code
 Grid search ensemble code for testing is provided i.e. model_gridsearch_load.py. It requires following files to run:
      -model_best_cnn.h5
      -model_best_lstm.h5
      -test_s33_Pos_51.fasta
      -test_s33_Neg_51.fasta
 The output will be our final result that we mentioned in our research paper.
 # Prediction for given test dataset (Procedure)
      - Download all rar files from model.part01 to model.part07 and keep in same folders.
      - Open model.part01 with compression tool like winrar and extract it. This will extract both model files from our 
        research, model_best_cnn.h5 and model_best_lstm.h5.
      - Download test datasets, test_s33_Pos_51.fasta and test_s33_Neg_51.fasta, and python code model_gridsearch_load.py.
        Keep them in same folder as model files.
      - Run model_gridsearch_load.py and you will get output mentioned in our research.
 # Prediction for your dataset
 The format should be same as the test dataset which is basically FASTA format. This model works for window size 51 only. 
 # Contact 
 Feel free to contact us if you need any help : dukka.kc@wichita.edu
