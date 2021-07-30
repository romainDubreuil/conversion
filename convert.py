import numpy as np

SPOT_NUM = 64      # number of spots
PEAKS_PER_SPOT = 3 # 3 peaks per spot

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

## Load MMI data and compute MZI

# load mmi text file
mmi_all = np.loadtxt('mmi_all.txt', dtype = 'float')

# oad mzi text file
sensorgram_from_mzi = np.loadtxt('mzi_all.txt', dtype = 'float')

# Calculate the sensorgram from the mmi
mmi_reshaped = np.reshape(mmi_all, [-1, SPOT_NUM, PEAKS_PER_SPOT])
sensorgram_from_mmi = computeSensorgram(mmi_reshaped)

np.savetxt('MZI_from_MMI.txt',sensorgram_from_mmi,fmt='%.8f')

# Compute error between MMI conversion and direct reading in sqlite data
error = sensorgram_from_mzi - sensorgram_from_mmi
error_mean = np.mean(np.abs(np.stack(error)))
error_std = np.std(np.abs(np.stack(error)))
with open('conversion_error.txt', 'w') as file:
	file.write(f'Mean of error: {error_mean}\n')
	file.write(f'Std of error: {error_std}')






