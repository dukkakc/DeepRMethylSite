# DeepRMethylSite: Prediction of Arginine Methylation in Proteins using Deep Learning

DeepRMethylSite provides a deep-learning method for Arginine Methylation sites prediction in proteins. It is implemented using Keras (version 2.2.4) and Tensorflow (version 1.15) backend both in Windows and Linux OS. 

# Pre-requisites
  Python 3.6<br/>
  Tensorflow (>version 1.15) - (Optional: Tensorflow GPU for GPU based machine)<br/>
  Keras (>version 2.2.4) - (Optional: Keras GPU for GPU based machine)<br/>
  Numpy <br/>
  Biopython <br/>
  Sklearn <br/>
  Imblearn <br/>
  Winrar or any compression program to open .rar file
  
 # Running on CPU or GPU
 To run in CPU installation of Tensorflow and Keras will suffice. However, to run in GPU, further Tensorflow-gpu and keras-gpu must be installed. Tensorflow GPU and Keras GPU version utilizes cuda cores in our GPU (in our case NVIDIA 2080 TI) for faster training time. However, running in GPU is not mandatory.
 
 # Dataset
 Dataset are in fasta format. Both training and testing datasets are provided which are independent (one does not include others).
 Training dataset for positive and negative are train_s33_Pos_51.fasta and train_s33_Neg_51.fasta respectively. Testing dataset for positive and negative are test_s33_Pos_51.fasta and test_s33_Neg_51.fasta respectively. Training dataset is made available so that future models can be train for the comparison purpose.
 # Model
 The best trained model for both CNN and LSTM (used in our final results) have also been included. The model_best_cnn.h5 is the best trained model for CNN and model_best_lstm.h5 is the best trained model for LSTM respectively. 
 # Code
 Grid search ensemble code for testing is provided i.e. model_gridsearch_load.py. It requires following files to run:
      -model_best_cnn.h5
      -model_best_lstm.h5
      -test_s33_Pos_51.fasta
      -test_s33_Neg_51.fasta
 The output will be our final result that we mentioned in our research paper.
 # Prediction for given test dataset (Procedure)
      - Download all rar files from model.part01 to model.part07 and keep in the same folder.
      - Open model.part01 with compression tool like winrar and extract it. This will extract both model files from our 
        research, model_best_cnn.h5 and model_best_lstm.h5.
      - Download test datasets, test_s33_Pos_51.fasta and test_s33_Neg_51.fasta, and python code model_gridsearch_load.py.
        Keep them in the same folder as model files.
      - Run model_gridsearch_load.py and you will get output mentioned in our research.
 # Prediction for your dataset
 The format should be same as the test dataset which is basically a FASTA format. This model works for window size 51 only meaning for the residue of your interest you should provide 25 resiudes downstream and 25 residues upstream. e.g. if you want to predict whether the 'Arginine' residue in Position 735 in protein Q4KWH8 is methylated or not, the input file should contain 25 residues upstream of R and 25 residues downstream of R.
 
 The general format for your dataset should be:

>sp|Q4KWH8|PLCH1_HUMAN%730%755
PKKQLILKVISGQQLPKPPDSMFGDRGEIIDPFVEVEIIGLPVDCCKDQTR

 # Contact 
 Feel free to contact us if you need any help : dukka.kc@wichita.edu
 
