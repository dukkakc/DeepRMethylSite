import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Lambda, LSTM
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import numpy as np
from Bio import SeqIO
from numpy import array
from numpy import argmax
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.backend import expand_dims
import matplotlib.pyplot as plt
from keras.regularizers import l1, l2
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.metrics import accuracy_score
from numpy import tensordot
from numpy.linalg import norm
from scipy.optimize import differential_evolution
from keras.models import load_model
from itertools import product
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix


r_test_x = []
r_test_y = []
r_test_x2 = []
r_test_y2 = []
posit_1 = 1;
negat_0 = 0;
win_size = 51 # actual window size
win_size1 = 39
win_size2 = 21
num_classes = 2
n_models = 2
cut_off1 = int((51 - win_size1)/2)
cut_off2 = int((51 - win_size2)/2)

# define universe of possible input values
alphabet = 'ARNDCQEGHILKMFPSTWYV-'
# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

# Test Dataset for CNN -------------------------------------------------------------
#for positive sequence
def innertest3():
    #Input
    data = seq_record.seq
    data = data[cut_off1:-cut_off1]
    #rint(data) 
    # integer encode input data
    for char in data:
        if char not in alphabet:
            return
    integer_encoded = [char_to_int[char] for char in data]
    r_test_x2.append(integer_encoded)
    r_test_y2.append(posit_1)
for seq_record in SeqIO.parse("test_s33_Pos_51.fasta", "fasta"):
    innertest3()
#for negative sequence
def innertest4():
    #Input
    data = seq_record.seq
    data = data[cut_off1:-cut_off1]
    #print(data) 
    # integer encode input data
    for char in data:
        if char not in alphabet:
            return
    integer_encoded = [char_to_int[char] for char in data]
    r_test_x2.append(integer_encoded)
    r_test_y2.append(negat_0)
for seq_record in SeqIO.parse("test_s33_Neg_51.fasta", "fasta"):
    innertest4()
# Changing to array (matrix)    
r_test_x2 = array(r_test_x2)
r_test_y2 = array(r_test_y2)


# Balancing test dataset
# Testing Data Balancing by undersampling####################################
rus = RandomUnderSampler(random_state=7)
x_res4, y_res4 = rus.fit_resample(r_test_x2, r_test_y2)
#Shuffling
r_test_x2, r_test_y2 = shuffle(x_res4, y_res4, random_state=7)
r_test_x2 = np.array(r_test_x2)
r_test_y2 = np.array(r_test_y2)


#-Test Dataset for LSTM----------------------------------------
#for positive sequence
def innertest1():
    #Input
    data = seq_record.seq
    data = data[cut_off2:-cut_off2]
    #rint(data) 
    # integer encode input data
    for char in data:
        if char not in alphabet:
            return
    integer_encoded = [char_to_int[char] for char in data]
    r_test_x.append(integer_encoded)
    r_test_y.append(posit_1)
for seq_record in SeqIO.parse("test_s33_Pos_51.fasta", "fasta"):
    innertest1()
#for negative sequence
def innertest2():
    #Input
    data = seq_record.seq
    data = data[cut_off2:-cut_off2]
    #print(data) 
    # integer encode input data
    for char in data:
        if char not in alphabet:
            return
    integer_encoded = [char_to_int[char] for char in data]
    r_test_x.append(integer_encoded)
    r_test_y.append(negat_0)
for seq_record in SeqIO.parse("test_s33_Neg_51.fasta", "fasta"):
    innertest2()
# Changing to array (matrix)    
r_test_x = array(r_test_x)
r_test_y = array(r_test_y)


# Balancing test dataset
# Testing Data Balancing by undersampling####################################
rus = RandomUnderSampler(random_state=7)
x_res3, y_res3 = rus.fit_resample(r_test_x, r_test_y)
#Shuffling
r_test_x, r_test_y = shuffle(x_res3, y_res3, random_state=7)
r_test_x = np.array(r_test_x)
r_test_y = np.array(r_test_y)
############################################################################



###############################################################################################################################################
# Ensemble Part


def ensemble_final_pred(members, weights, testX, testX2):
	# make predictions
    yhats = []
    yhats.append(array(members[0].predict(testX)))
    yhats.append(array(members[1].predict(testX2)))
    yhats = array(yhats)
	# weighted sum across ensemble members
    summed = tensordot(yhats, weights, axes=((0),(0)))
	# argmax across classes
	#result = argmax(summed, axis=1)
    return summed

# Load models 

models = [load_model('model_best_lstm.h5'),load_model('model_best_cnn.h5')]
for i in models:
    print(i.summary())

# grid search weights
#weights = grid_search(models, x_test, y_test)
#weights = grid_search(models, r_test_x, r_test_y, r_test_x2, r_test_y2)
weights = [0.16666667,0.83333333]
# Independent Test


Y_pred = ensemble_final_pred(models, weights, r_test_x, r_test_x2)
t_pred2 = Y_pred[:,1]
Y_pred = (Y_pred > 0.5)
y_pred1 = [np.argmax(y, axis=None, out=None) for y in Y_pred]
y_pred1 = np.array(y_pred1)

print("Matthews Correlation : ",matthews_corrcoef(r_test_y, y_pred1))
print("Confusion Matrix : \n",confusion_matrix(r_test_y, y_pred1))

# For sensitivity and specificity
sp_1, sn_1 = confusion_matrix(r_test_y, y_pred1)
sp_2 = sp_1[0]/(sp_1[0]+sp_1[1])
sn_2 = sn_1[1]/(sn_1[0]+sn_1[1])
# ROC

fpr, tpr, _ = roc_curve(r_test_y, t_pred2)
roc_auc = auc(fpr, tpr)
print("AUC : ", roc_auc)
print(classification_report(r_test_y, y_pred1))

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for ST')
plt.legend(loc="lower right")
plt.show()

print("Specificity = ",sp_2, " Sensitivity = ",sn_2)

