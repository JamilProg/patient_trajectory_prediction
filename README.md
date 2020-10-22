# Step 0 : Python environment
All of these scripts were ran with Python 3.7.

# Step 1 : Cleaning data from MIMIC III's NoteEvents.csv (data cleaning)
1.1 Move to data_cleaning folder.

1.2 Run noteEvents_preproc.py (with MIMIC III's NOTEEVENTS.csv as input) - it takes about 4 hours to finish, and generates a preprocessed text (output.csv).

1.3 Run MIMIC_smart_splitter.py (with output.csv as input) : splits the preprocessed text into files of 50 Mb without cutting any note - it should take about 1 hour.

1.4 At this step, we have a new folder called "data" which contains two folders. The first one (chunkssmall) contains all files and the other one is empty.

# Step 2 : CUI Recognizer with QuickUMLS 

2.1 Install QuickUMLS, see : https://github.com/Georgetown-IR-Lab/QuickUMLS - at the end, you should have a QuickUMLS folder, as follow :

![Alt text](miscellaneous/QU_repo.png?raw=true "QuickUMLS Repository tree structure")

2.2 Put the "data" folder generated in step 1.3, and the installed "QuickUMLS" folder in concept_annotation folder.

2.3 Once you're in concept_annotation folder, run quickUMLS_getCUI.py (this process takes days to finish, depending on your parameters).

Parameters are:

--t : Float which is QuickUMLS Threshold, should be between 0 and 1 (default 0.9).

--TUI : String which represents the TUI List filter, either "Alpha" or "Beta" (default Beta).

2.4 Concatenate the multiple outputs to make one final file. For that, move to "data/outputchunkssmall" and run the 4th and last command mentionned in : useful_commands.txt

2.5 Run quickumls_processing.py with the concatenated output as input (output of the previous step).

A new file is generated, the data is ready for Deep Learning !

# Step 3 : Deep Learning

## Step 3.1 : Data preparation

1.1 Put the data file in "PyTorch_scripts/any target task/".

1.2 Run 01_data_preparation.py

Parameters are:

--admissions_file : path to the MIMIC III's ADMISSIONS.csv file.

--diagnoses_file : path to the MIMIC III's DIAGNOSES_ICD.csv file.

--notes_file : path to the data file.

--output_file_name : name of the output (of your choice).

1.3 A npz file is generated, your data is ready for training!

## Step 3.2A : Diagnoses prediction

## Step 3.2B : Mortality prediction

## Step 3.2C : Readmission prediction

https://github.com/JamilProg/script_preproc_MIMIC/blob/master/README.md
