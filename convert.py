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


def read_mmi_sensorgram(data):

	# MMIs
	mmi_name = ['a', 'b', 'c']
	mmi_all = []
	for name in mmi_name:
		temp = np.array(data['timeseries'][f'osense:mmi.{name}'])
		for i, col in enumerate(temp[:,0]):
			if i==0:
				mmi = col
			else:
				mmi = np.vstack((mmi, col))
		mmi_all.append(mmi)
	mmi_all = np.stack(mmi_all, axis=2).astype(float)
	time = np.array(data['timeseries']['osense:mmi.a'])[:,1].astype(float)

	# Convert time to second
	time_converted = np.array([datetime.fromtimestamp(time[k]/1e3) for k in range(time.shape[0])]) 	# convert UTC timestamp in nanosecond to datetime object
	time = np.array([i.total_seconds() for i in (time_converted - time_converted[0])])

	# Sensorgram
	sensorgram  = np.array(data['data']).reshape((time.shape[0],SPOT_NUM))

	# Create dataframe
	spot_names = [nameOfSpot(spot)[0:2] for spot in range(SPOT_NUM)]
	sensorgram_df = pd.DataFrame(index=time, columns=spot_names, data=sensorgram)

	mmi_df = pd.DataFrame(index=time)
	for i in range(mmi_all.shape[1]*mmi_all.shape[2]):
		spot = i//3
		phase = i%3

		mmi_df[nameOfSpot(spot, phase)] = mmi_all[:, spot, phase]

	return Munch(mmi=mmi_df, sensorgram=sensorgram_df, time=time)

def computeSensorgram(mmi):

	def computeIQ(mmi):
		p1 = mmi[:,:,0]
		p2 = mmi[:,:,1]
		p3 = mmi[:,:,2]
		it = 2 * p2 - p1 - p3
		qt = np.sqrt(3)*(p1 - p3)
		iq = np.stack((it,qt), axis=-1) # dimensions: (time, spots, [0]=i [1]=q)
		return iq

	def computeAngles(iq):
		it = iq[:,:,0]
		qt = iq[:,:,1]
		angle = -np.arctan2(qt, it)
		return angle

	def removeSensorgramJumps(angles):
		'''
		correction of 2pi angle jump
		'''
		frame_count = angles.shape[0]
		max_angle_jump = np.pi
		corrected_angles = np.copy(angles)
		for spot in range(angles.shape[1]):
			for t in range(1, frame_count):
				if (corrected_angles[t,spot] > corrected_angles[t-1,spot] + max_angle_jump):
					corrected_angles[t:,spot] -= 2*np.pi
				
				else:
					if (corrected_angles[t,spot] < corrected_angles[t-1,spot] - max_angle_jump):
						corrected_angles[t:,spot] += 2*np.pi
		return corrected_angles


	iq = computeIQ(mmi)

	sensorgram = computeAngles(iq)
	sensorgram = removeSensorgramJumps(sensorgram)	# remove discontinuity caused by atan2
	
	offset = sensorgram[0,:]
	sensorgram = sensorgram - offset
	return sensorgram


def nameOfSpotWithType(fspType, spot, phase=None):
	if fspType == "Fsp16":
		# A0 (15) A1 (11) A2 (7) A3 (3)
		# B0 (14) B1 (10) B2 (6) B3 (2)
		# C0 (13) C1 (9)  C2 (5) C3 (1)
		# D0 (12) D1 (8)  D2 (4) D3 (0)
		col = spot // 4
		row = spot % 4
	else:
		# A0 (60) A1 (61) A2 (62) A3 (63)
		# ...
		# P0 (0)  P1 (2)  P2 (3)  P3 (4)
		row = 15 - spot // 4
		col = spot % 4
	return "%s%d[%d%s]" % (
		chr(row+ord('A')),col,
		spot,chr(phase+ord('a')) if phase is not None else '')


def nameOfSpot(spot, phase=None):
	return nameOfSpotWithType('Fsp64', spot, phase)


## Load MMI data and compute MZI

# construct data object


# load text files
mmi_a_loaded = np.loadtxt('MMI_a.txt', dtype = 'float')
mmi_b_loaded = np.loadtxt('MMI_b.txt', dtype = 'float')
mmi_c_loaded = np.loadtxt('MMI_c.txt', dtype = 'float')
timestamp = np.loadtxt('time_mmi.txt', dtype = 'float')


# build MMI object in data_rec
mmi_a_rec = []
mmi_b_rec = []
mmi_c_rec = []
for i in range(len(timestamp)):
	mmi_a_rec.append([mmi_a_loaded[i],timestamp[i]])
	mmi_b_rec.append([mmi_b_loaded[i],timestamp[i]]) 
	mmi_c_rec.append([mmi_c_loaded[i],timestamp[i]]) 

data_rec={
	"timeseries": 
		{
		"osense:mmi.a":mmi_a_rec,
		"osense:mmi.b":mmi_b_rec,
		"osense:mmi.c":mmi_c_rec
		}
	,
	"data":[]
}

sensorgram_from_mzi = np.loadtxt('data_from_sqlite.txt', dtype = 'float')
data_rec["data"] = sensorgram_from_mzi

# data_rec object is done and can be processed
data = read_mmi_sensorgram(data_rec)

time = data.time
mmi = data.mmi
sensorgram_from_sqlite = data.sensorgram

# Calculate the sensorgram from the mmi in Sqlite
mmi_reshaped = np.reshape(mmi.values, [mmi.shape[0], SPOT_NUM, PEAKS_PER_SPOT])
sensorgram_from_mmi = computeSensorgram(mmi_reshaped)

np.savetxt('MZI_from_MMI.txt',sensorgram_from_mmi,fmt='%.8f')

sensorgram_from_mmi = pd.DataFrame(index=time, columns=sensorgram_from_sqlite.columns, data=sensorgram_from_mmi)	# numpy to dataframe

# Compute error between MMI conversion and direct reading in sqlite data
error = sensorgram_from_sqlite - sensorgram_from_mmi
error_mean = error.stack().abs().mean() 
error_std = error.stack().abs().std()
file = open('conversion_error.txt', 'w')
file.write(f'Mean of error: {error_mean}\n')
file.write(f'Std of error: {error_std}')
file.close()






