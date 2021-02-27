# Step 0 : Python environment
- All of these scripts were ran with Python 3.7.
- PyTorch version 1.5.0
- Cuda version 10.2
- GPU Quadro P6000

# Step 1 : Cleaning data from MIMIC III's NoteEvents.csv (data cleaning)
1.1 Move to data_cleaning folder.

1.2 Run noteEvents_preproc.py (with MIMIC III's NOTEEVENTS.csv as input) - it takes about 4 hours to finish, and generates a preprocessed text (output.csv).

1.3 Run MIMIC_smart_splitter.py (with output.csv as input) : splits the preprocessed text into files of 50 Mb without cutting any note - it should take about 1 hour.

1.4 At this step, we have a new folder called "data" which contains two folders. The first one (chunkssmall) contains all files and the other one is empty.

# Step 2 : CUI Recognizer with QuickUMLS (concept extraction)

2.1 Install QuickUMLS, see : https://github.com/Georgetown-IR-Lab/QuickUMLS - at the end, you should have a QuickUMLS folder, as follow :

![Alt text](miscellaneous/QU_repo.png?raw=true "QuickUMLS Repository tree structure")

2.2 Put the "data" folder generated in step 1.3, and the installed "QuickUMLS" folder in concept_annotation folder.

2.3 Once you're in concept_annotation folder, run quickUMLS_getCUI.py (if your machine is able to run about 25-30 threads, this process takes between hours to 3 days to finish, depending on the chosen parameters).

Parameters are:

* --t : Float which is QuickUMLS Threshold, should be between 0 and 1 (default --t=0.9).
* --TUI : String which represents the TUI List filter, either "Alpha" or "Beta" (default --TUI=Beta).

Note: Both TUI lists are available in TUI_Lists.pdf file in the root of the repository.

2.4 Concatenate the multiple outputs to make one final file. For that, move to "data/outputchunkssmall" and run the 4th and last command mentioned in : useful_commands.txt

2.5 Run quickumls_processing.py with the concatenated output as input (output of the previous step).

A new file is generated, the data is ready for Deep Learning !

# Step 3 : Deep Learning (PyTorch scripts)

## Step 3.1 : Data preparation

1.1 Put the data file in "PyTorch_scripts/(any_target_task)/".

1.2 Run 01_data_preparation.py (or 01_data_prep_mortality.py / 01_data_prep_readmission.py depending on the chosen task).

Parameters are:

* --admissions_file : path to the MIMIC III's ADMISSIONS.csv file.
* --diagnoses_file : path to the MIMIC III's DIAGNOSES_ICD.csv file.
* --notes_file : path to the data file.

1.3 A npz file (two for mortality_prediction) is generated, your data is ready for training!

## Step 3.2A : Diagnoses prediction

Option 1 - FFN: Run 02_FFN_diagprediction.py (K-Fold Crossvalidation)

Optional arguments are:

* --withCCS : add --withCCS=1 if you want to add CCS feature in the input (CCS one-hot concatenated with CUI one-hot)
* --hiddenDimSize : size of the hidden layer
* --batchSize : size of batches
* --nEpochs : number of epochs
* --lr : learning rate
* --dropOut : drop-out probability in the last layer

Option 2 - RNN (NOT k-fold crossvalidation script because too long and heavy): train by running 02_GRU_train_GPU.py for GRU (or 02_LSTM_train_GPU.py for LSTM)

Then, test by running 03_GRU_test.py for GRU (or 03_LSTM_test.py for LSTM).

Optional arguments for both RNN models are:

* --withCCS : add --withCCS=1 if you want to add CCS feature in the input (CCS one-hot concatenated with CUI one-hot) [only training script]
* --hiddenDimSize : size of the hidden layer [both training script and testing script]
* --batchSize : size of batches [both training script and testing script]
* --nEpochs : number of epochs [only training script]
* --lr : learning rate [only training script]
* --dropOut : drop-out probability in the last layer [both training script and testing script]

## Step 3.2B : Mortality prediction

In mortality_prediction folder, you can train and test a model (K-Fold Crossvalidation) whose architecture is :

Option 1) Fully-connected (02_FFN_mortality.py)

Option 2) RNN with Gated Recurrent Unit cells (02_GRU_mortality.py)

Optional arguments for both models are:

* --withCCS : add --withCCS=1 if you want to add CCS feature in the input (CCS one-hot concatenated with CUI one-hot)
* --hiddenDimSize : size of the hidden layer
* --batchSize : size of batches
* --nEpochs : number of epochs
* --lr : learning rate
* --dropOut : drop-out probability in the last layer

## Step 3.2C : Readmission prediction

For readmission prediction, it is mainly the same method and arguments as mortality_prediction, but in readmission_prediction folder.

https://github.com/JamilProg/script_preproc_MIMIC/blob/master/README.md
