from pandas.core.reshape import reshape
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from array import array
from IPython.display import display


def computeIQ(peaks):
    phaseShift = np.array(peaks).reshape(-1, SPOT_COUNT, 3)
    p1 = phaseShift[:,:,0]
    p2 = phaseShift[:,:,1]
    p3 = phaseShift[:,:,2]

    it = 2 * p2 - p1 - p3
    qt = np.sqrt(3) * (p1 - p3)

    return np.array([it[0,:], qt[0,:]])

def computePhase(iq):
    it = iq[0]
    qt = iq[1]
    return -np.arctan2(qt, it)

def load_peaks(index):
    # return an array of 192 elements
    MMI= np.concatenate((np.array(mmi_a_loaded[index]),np.array(mmi_b_loaded[index]), np.array(mmi_c_loaded[index] )), axis=0)   
    return MMI

def main(dataset_length):

    output = []

    # reshape first point
    MMI = load_peaks(0)
    arr2D = np.reshape(MMI, (3, 64)).T
    peaks = np.reshape(arr2D, 192, order='C')
    iq = computeIQ(peaks)
    phase = computePhase(iq)
    firstPhase    = phase
    previousPhase = phase
    k = np.zeros(phase.shape[0])

    # for loop to compute the data
    for i in range(dataset_length):
        MMI = load_peaks(i)
        arr2D = np.reshape(MMI, (3, 64)).T
        peaks = np.reshape(arr2D, 192, order='C')
        iq = computeIQ(peaks)
        phase = computePhase(iq)
        phaseDiff = phase - previousPhase
        previousPhase = phase
        
        # 2pi jumps correction
        jumps = np.abs(phaseDiff) > np.pi
        if jumps.any():
            k -= np.sign(phaseDiff)*jumps
        sensogram = 2*k*np.pi + phase - firstPhase
        output.append(sensogram)

    return output


SPOT_COUNT = 64

# load text files
mmi_a_loaded = np.loadtxt('MMI_a.txt', dtype = 'float')
mmi_b_loaded = np.loadtxt('MMI_b.txt', dtype = 'float')
mmi_c_loaded = np.loadtxt('MMI_c.txt', dtype = 'float')

########################################################################################
# Run main function and save data as text (data saving as MZI_from_MMI is not required)
dataset_length = len(mmi_a_loaded)
sensorgram_from_mmi = main(dataset_length)
np.savetxt('MZI_from_MMI.txt',sensorgram_from_mmi,fmt='%.8f')

# compare the 2 files : from sqlite and computed MZI.
# Compute error between MMI conversion and direct reading in sqlite data

sensorgram_from_sqlite = np.loadtxt('data_from_sqlite.txt', dtype = 'float')

error = sensorgram_from_sqlite - np.reshape(sensorgram_from_mmi, (1, len(sensorgram_from_sqlite)))

error_mean = np.mean(np.abs(error)) 
error_std = np.mean(np.std(error)) 
file = open('conversion_error.txt', 'w')
file.write(f'Mean of error: {error_mean}\n')
file.write(f'Std of error: {error_std}')
file.close()

####################################
# uncomment to plot if wanted to see the curves
####################################
# plot_name = ['plot_'+ str(x) for x in range(64)]
# df = pd.DataFrame.from_records(output)
# df.columns = plot_name
# ax = plt.gca()
# for index in plot_name:
#     df.plot(kind='line',y=index,ax=ax) 
# plt.show()

