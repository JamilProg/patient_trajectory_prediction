All of these scripts were ran with Python3.

# Step 1 : Cleaning data from MIMIC III's NoteEvents.csv (data cleaning)
1.0 Move to data_cleaning folder.

1.1 Run noteEvents_preproc.py (with MIMIC III's NOTEEVENTS.csv as input) - this process takes about 4 hours to finish, and generates a preprocessed text (output.csv).

1.2 Run MIMIC_smart_splitter.py (with output.csv as input) : splits the preprocessed text into files of 50 Mb without cutting any note.

1.3 At this step, we have a new folder called "data" which contains two folders. The first one (chunkssmall) contains all files and the other one is empty.

# Step 2 : CUI Recognizer with QuickUMLS 

2.1 Install QuickUMLS, see : https://github.com/Georgetown-IR-Lab/QuickUMLS - at the end, you should have a QuickUMLS folder.

2.2 Put the "data" folder generated in step 1.3, and the installed "QuickUMLS" folder in concept_annotation folder.

2.3 Once you're in concept_annotation folder, run quickUMLS_getCUI.py (this process takes days to finish, depending on your parameters).
[Explain the parameters]

2.4 Concatenate the multiple outputs to make one final file, see the command here : useful_commands.txt

2.5 Run quickumls_processing.py with the concatenated outputs as input.

The data is ready !

# Step 3 : Deep Learning

(Work is in progress)

https://github.com/JamilProg/script_preproc_MIMIC/blob/master/README.md
