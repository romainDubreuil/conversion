Conversion task : 

1/ run mmi2mzi_generator.py

2/ See that the error between the input file (data_from_sqlite.txt) and the generated output file (MZI_from_MMI.txt) is computed in the conversion_error.txt

3/ mean error and std error should be inferior to 10-e7 as a request to validate the conversion. 
Mean of error: 5.791429022163692e-08
Std of error: 8.48607493065224e-08
from the python code

**** note ****
sqlite_to_text_files.py is not used (aryballe tool only)