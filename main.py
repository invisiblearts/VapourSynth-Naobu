import os
import math
import functools
import sys

import vapoursynth
import caffe
import numpy


def FrameWaifu(n, f, net, block_h, block_w):
    assert (isinstance(f, vapoursynth.VideoFrame))
    assert (isinstance(net, caffe.Net))
    pad_dim = 7
    net.blobs['input'].reshape(1, 1, 2 * pad_dim + block_h, 2 * pad_dim+block_w)
    frame = f.copy()
    frameArr = numpy.array(frame.get_read_array(0), copy=False)

    orig_width = frameArr.shape[1]
    orig_height = frameArr.shape[0]
    padded_width = math.ceil(orig_width / block_w) * block_w
    padded_height = math.ceil(orig_height / block_h) * block_h
    padded_h1, padded_h2, padded_w1, padded_w2 = math.floor((padded_height - orig_height) / 2), \
                                                 math.ceil((padded_height - orig_height) / 2), \
                                                 math.floor((padded_width - orig_width) / 2), \
                                                 math.ceil((padded_width - orig_width) / 2)

    frameArr = numpy.pad(frameArr, ((padded_h1, padded_h2), (padded_w1, padded_w2)), 'reflect')
    inList = []
    outList = []
    for i in numpy.hsplit(frameArr, int(padded_width / block_w)):
        inList.append(numpy.vsplit(i, int(padded_height / block_h)))
    for i in inList:
        tmp = []
        for j in i:
            j = numpy.pad(j, pad_dim, 'reflect')
            net.blobs['input'].data[...] = j
            net.forward()
            j = numpy.array(net.blobs['conv7'].data, copy=True)
            tmp.append(j)
        outList.append(tmp)
    tmpList = []
    for i in outList:
        for j in i:
            j = numpy.squeeze(j)
        tmpList.append(numpy.concatenate(i, -2))
    outArr = numpy.concatenate(tmpList, -1)
    outArr = numpy.squeeze(outArr)
    sys.stderr.write(str(outArr.shape))
    outArr = outArr[padded_h1:-padded_h2, padded_w1:-padded_w2]
    plane = frame.get_write_array(0)
    for i in range(0, orig_width):
        for j in range(0, orig_height):
            plane[j, i] = outArr[j, i]
    return frame


def Waifu2x(clip, block_w=128, block_h=128, mode=0):
    vs_core = vapoursynth.get_core()
    func_name = 'Waifu2x'
    scriptDir = os.path.split(os.path.realpath(__file__))[0]
    modelDir = ['anime_style_art', 'anime_style_art_rgb']
    model = ['scale2x.caffemodel', 'noise1.caffemodel', 'noise2.caffemodel']
    proto = 'srcnn.prototxt'

    if not isinstance(clip, vapoursynth.VideoNode):
        raise TypeError(func_name + r': "clip" must be a clip!')

    if not isinstance(block_w, int) or not isinstance(block_h, int) or block_w <= 0 or block_h <= 0:
        raise ValueError(func_name + r': "block_w/block_h" must be positive integers!')

    sFormat = clip.format
    sColorFamily = sFormat.color_family
    sWidth = clip.width
    sHeight = clip.height

    if sFormat.bytes_per_sample is not 4 or sFormat.sample_type is not vapoursynth.FLOAT:
        raise ValueError(func_name + r': Sample type/depth not supported! Must be 32bit float!')

    if sColorFamily == vapoursynth.YUV:
        modelPath = scriptDir + os.sep + modelDir[0] + os.sep + model[mode]
        protoPath = scriptDir + os.sep + modelDir[0] + os.sep + proto
    elif sColorFamily == vapoursynth.RGB:
        modelPath = scriptDir + os.sep + modelDir[1] + os.sep + model[mode]
        protoPath = scriptDir + os.sep + modelDir[1] + os.sep + proto
        raise ValueError(func_name + r': Currently RGB is not supported. Sorry.')
    else:
        raise ValueError(func_name + r': Color family unsupported! Must be RGB/YUV!')

    # Net Initialization
    net = caffe.Net(protoPath, modelPath, caffe.TEST)

    if mode == 0:
        pre1 = vs_core.resize.Bicubic(clip, 2 * clip.width, 2 * clip.height)
    else:
        pre1 = clip

    returnClip = vs_core.std.ModifyFrame(pre1, pre1,
                                         functools.partial(FrameWaifu,
                                                           net=net, block_h=block_h, block_w=block_w))

    return returnClip
