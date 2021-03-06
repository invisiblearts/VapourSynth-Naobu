import vapoursynth as vs
import h5py
import mvsfunc as mvf
import numpy as np
import gc
import random

def get_data_from_frame(frame, dim):
    assert isinstance(frame, vs.VideoFrame)
    assert isinstance(dim, int)
    arr = frame.get_read_array(0)
    returnList = []
    for i in range(0, 10):
        for j in range(5, 7):
            returnList.append(arr[j * dim : (j + 1) * dim, i * dim : (i + 1) * dim])
    del arr
    return returnList

prefix = 'train'
suffix = '.h5'
labelOutput = prefix + '_label' + suffix
dataOutput = prefix + '_data' + suffix

data = '00003.mp4' # Edit it to your video source
label = '00003.m2ts'
dataDim = 80
padDim = 6 # Actually 6 on each side
paddedDim = dataDim + 2 * padDim


# Get source and extract Y
core = vs.get_core()
dataClip = mvf.Depth(core.lsmas.LWLibavSource(data), 32).std.ShufflePlanes(0, vs.GRAY)
labelClip = mvf.Depth(core.lsmas.LWLibavSource(label), 32).std.ShufflePlanes(0, vs.GRAY)
assert isinstance(dataClip, vs.VideoNode)
assert isinstance(labelClip, vs.VideoNode)
assert labelClip.num_frames == dataClip.num_frames
assert dataClip.height == labelClip.height
assert dataClip.width == labelClip.width
frameNum = dataClip.num_frames
frame_h = dataClip.height
frame_w = dataClip.width

sampleFrameNum = 5000
samplePerFrame = 10
sampleNum = sampleFrameNum * samplePerFrame
assert labelClip.num_frames >= sampleFrameNum

# Prepare HDF5 database
labelFile = h5py.File(labelOutput, 'w')
dataFile = h5py.File(dataOutput, 'w')
datafile.create_dataset('data', (sampleNum, 1, paddedDim, paddedDim), 'single')
labelfile.create_dataset('label', (sampleNum, 1, dataDim, dataDim), 'single')

startloc = 0

# Get data from clip and write it to HDF5
numFrames = labelClip.num_frames
i = 0
currentSample = 0
while i < sampleFrameNum:
    print(str(i) + '\n')
    currentFrame = random.randint(0, numFrames - 1)
    currentDataFrame = dataClip.get_frame(currentFrame)
    currentLabelFrame = dataClip.get_frame(currentFrame)
    dataSubList = get_data_from_frame(currentDataFrame, paddedDim)
    labelSubList = get_data_from_frame(currentLabelFrame, paddedDim)
    m = 0
    while m < samplePerFrame:
        current_num = i * samplePerFrame + m
        datafile['data'][current_num] = dataSubList[m]
        labelfile['label'][current_num] = labelSubList[m][padDim:-padDim, padDim:-padDim]
        m += 1
    i += 1
    del currentDataFrame, currentLabelFrame, dataSubList, labelSubList
    gc.collect()
