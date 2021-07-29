import numpy as np

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

# data_from_sqlite = read_record_in_sqlite(database, record_id)
# with open('record.json', 'w') as f:
#     json.dump(data_from_sqlite, f)

with open('record.json', 'r') as f:
    data_from_sqlite = json.load(f)

# Write mmi a, b and c
mmi_a = data_from_sqlite["timeseries"]["osense:mmi.a"]
mmi_b = data_from_sqlite["timeseries"]["osense:mmi.b"]
mmi_c = data_from_sqlite["timeseries"]["osense:mmi.c"]


mmi_all = []
for (a_frame, _), (b_frame, _), (c_frame, _) in zip(mmi_a, mmi_b, mmi_c):
    mmi_line = []
    for a, b, c in zip(a_frame, b_frame, c_frame):
        mmi_line.extend([a, b, c])  # Extrnd is like append(a), apend(b), append(c), but in one line
    assert len(mmi_line) == 192
    mmi_all.append(mmi_line)  # Append a line with 192 values

np.savetxt('mmi_all.txt',mmi_all,fmt='%.18f')

# Write sqlite sensorgram
data_mzi = data_from_sqlite["data"]
data_mzi = np.reshape(data_mzi, [-1, 64])
np.savetxt('mzi_all.txt',data_mzi,fmt='%.18f')







