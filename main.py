import os
import math

import vapoursynth
import caffe
import numpy


def FrameWaifu(frame, net, block_h, block_w):
    assert(isinstance(frame, vapoursynth.VideoFrame))
    assert(isinstance(net, caffe.Net))
    frameArr = numpy.array(frame.get_read_array(0), copy=False)
    inList = []
    outList = []
    for i in numpy.hsplit(frameArr, int(frame.width / block_w)):
        inList.extend(numpy.vsplit(i, int(frame.height / block_h)))
    for i in inList:
        net.blobs['input'].data[...] = i
        outList.append(net.forward())


def Waifu2x(clip, block_w = 142, block_h = 142, mode = 0):
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

    padded_width = math.ceil(pre1.width / block_w) * block_w
    padded_height = math.ceil(pre1.height / block_h) * block_h

    # Pad image
    pre2 = vs_core.fmtc.resample(pre1, padded_width, padded_height, (pre1.width - padded_width) / 2,
                                (pre1.height - padded_height) / 2, padded_width, padded_height,
                                 planes=[3, 1, 1])

