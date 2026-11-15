GROUP EIC-11 - Wearable Sensor-Based Activity Recognition Using CNN-LSTM
=========================================================================
Members: Vidhaan Mathur (1025050137), Tanishka Yadav (1025050006),
         Harsimrat Kaur Kang (1025050134), Bhuvi Sharma (1025050031),
         Dev Kukreja (1025050126)

1. WHAT THIS CODE DOES
----------------------
AI_EST_Proj_Code.ipynb classifies six activities (walking, upstairs,
downstairs, sitting, standing, laying) from nine inertial-signal channels
(128 time steps each) of the UCI HAR dataset. It:
  1. Loads the nine raw signals and labels for the train and test groups.
  2. Plots the class distribution and one sample window per activity.
  3. Splits the 21 training subjects by SUBJECT into training and validation
     (4 validation subjects chosen with RandomState(42)).
  4. Z-score normalises each channel using TRAINING statistics only.
  5. Trains a baseline LSTM and a hybrid CNN-LSTM (Adam, batch size 64,
     max 60 epochs, EarlyStopping on val_loss, patience 8,
     restore_best_weights=True, loss = sparse categorical cross-entropy).
  6. Prints test accuracy for both models, the classification report and
     confusion matrix of the CNN-LSTM, and a validation-accuracy plot.

2. FOLDER CONTENTS
------------------
  AI_EST_Proj_Code.ipynb   main notebook (all code)
  README.txt               this file
  requirements.txt         Python dependencies

3. DATASET
----------
UCI "Human Activity Recognition Using Smartphones" (UCI HAR Dataset)
  https://archive.ics.uci.edu/dataset/240/humanactivityrecognitionusingsmartphones
Download the zip, extract it, and note the folder that directly contains
the subfolders "train" and "test" (usually ...\UCI HAR Dataset\UCI HAR Dataset).
The dataset is NOT modified by the code.

4. HOW TO RUN
-------------
  a) Install Python 3 and create an environment (optional but recommended):
         python -m venv venv
         venv\Scripts\activate          (Windows)   or   source venv/bin/activate
  b) Install dependencies:
         pip install -r requirements.txt
  c) Open AI_EST_Proj_Code.ipynb:
         jupyter notebook
  d) IMPORTANT - edit the ROOT variable in the FIRST cell. It currently points
     to the author's machine:
         ROOT = r"C:\Users\Vidhaan\Documents\UCI HAR Dataset\UCI HAR Dataset"
     Change it to the folder from step 3 on your computer.
  e) Run all cells in order (Kernel > Restart & Run All). Cell 1 loads data,
     the next cell plots and splits, the last cell trains and evaluates.
     Training takes a few minutes on a CPU.

5. REPRODUCIBILITY NOTE
-----------------------
Seeds are fixed (tf.keras.utils.set_random_seed(42) and RandomState(42) for
the validation-subject choice), but exact numbers can still differ slightly
between machines and TensorFlow versions, so results from a new run may not
match the report to the last digit.

6. OWN WORK VS EXTERNAL RESOURCES
---------------------------------
External: UCI HAR dataset (Anguita et al., ESANN 2013); the libraries
TensorFlow/Keras, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn; the
general CNN-LSTM idea from the literature (Hochreiter & Schmidhuber 1997;
Ordonez & Roggen 2016).
Own work: data loading/stacking, subject-wise split, normalisation, plots,
both model definitions, training/evaluation code and analysis.
No pre-trained weights are used.
