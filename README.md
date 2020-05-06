# script_preproc_MIMIC

Step 1 : Cleaning data from MIMIC III's NoteEvents.csv (data cleaning)

1.1 Run noteEvents_preproc.py (with NoteEvents.csv as input) - this process takes at least 2 hours to finish
1.2 MIMIC_smart_splitter.py : split the output of the previous process into files of 50 Mb without cutting any note.


Step 2.A : CUI Recognizer with QuickUMLS 

2.1 Install QuickUMLS, see : https://github.com/Georgetown-IR-Lab/QuickUMLS
2.2 Run quickUMLS_getCUI_parallel.py
2.3 Concatenate the multiple outputs, see : useful_commands.txt
2.4 Run quickumls_processing.py with the concatenated outputs as input
The data is ready !

Step 2.B : CUI Recognizer with MetaMap

2.1 Install MetaMap, see : https://metamap.nlm.nih.gov/Installation.shtml
2.2 Run the servers, see : useful_commands.txt
2.3 Call the tool to annotate, see : useful_commands.txt
2.4 Concatenate the multiple outputs, see : useful_commands.txt
2.5 Run metamap_processing.py with the concatenated outputs as input
The data is ready !

Step 3 : Deep Learning
(Work is in progress)
