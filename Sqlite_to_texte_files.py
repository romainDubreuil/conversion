from datetime import datetime
import numpy as np
import pandas as pd
from munch import Munch, munchify
import warnings
warnings.filterwarnings('ignore')



import sqlite3
import json



SPOT_NUM = 64      # number of spots
PEAKS_PER_SPOT = 3 # 3 peaks per spot


def read_record_in_sqlite(path, run_id):

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    query = f'''select record from record where id = '{run_id}'
	'''

    cur.execute(query)
    data = json.loads(cur.fetchone()[0])
    
    conn.close()
    return data

## Load MMI data and compute MZI

######## TO ADAPT IN YOUR CODE
database = r"C:\Users\RomainDubreuil\Aryballe\hw - General\200420-watch\algorithmics\210708_portageAlgo\210604_ECOSSE_fnf-solution_GOLD_sample_8.sqlite" # To adapt with your path
######## That was the only place to adapt.

record_id = '1622802740113_NOA221-100010-CS220-1145_fsp'
data_from_sqlite = read_record_in_sqlite(database, record_id)


# Write mmi a, b and c
mmi_a = data_from_sqlite["timeseries"]["osense:mmi.a"]
mmi_b = data_from_sqlite["timeseries"]["osense:mmi.b"]
mmi_c = data_from_sqlite["timeseries"]["osense:mmi.c"]

mmi_a_txt = []
mmi_b_txt = []
mmi_c_txt = []
time_mmi = []


for x in mmi_a:
    mmi_a_txt.append(x[0])
    time_mmi.append(x[1]) 
for x in mmi_b:
    mmi_b_txt.append(x[0])
    
for x in mmi_c:
    mmi_c_txt.append(x[0])
   

np.savetxt('MMI_a.txt',mmi_a_txt,fmt='%.8f')
np.savetxt('MMI_b.txt',mmi_b_txt,fmt='%.8f')
np.savetxt('MMI_c.txt',mmi_c_txt,fmt='%.8f')

np.savetxt('time_mmi.txt',time_mmi,fmt='%.8f')

# Write sqlite sensorgram
data_mzi = data_from_sqlite["data"]
np.savetxt('data_from_sqlite.txt',data_mzi,fmt='%.8f')







