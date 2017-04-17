from NatronGui import *
import os.path


def generarNodos():
    ShotCutDetectionNode    = app1.ShotCutDetection1
    reader                  = app1.Read1
    viewer                  = app.Viewer1
    merger                  = app.createNode("net.sf.openfx.MergePlugin")
    viewer.connectInput(0, merger)
    SDA                     = ShotCutDetectionNode.getParam("SDA")
    thresholdParam          = ShotCutDetectionNode.getParam("Threshold")
    threshold               = thresholdParam.getValueAtTime(0)
    firstFrame              = reader.getParam("firstFrame").get()
    lastFrame               = reader.getParam("lastFrame").get()
    readerPositionX         = reader.getPosition()[0]
    readerPositionY         = reader.getPosition()[1]
    merger.setPosition(readerPositionX,readerPositionY+350) # down it goes
    viewer.setPosition(readerPositionX,readerPositionY+600) # down it goes
    amountOfNodes           = 0
    cuts = [0]
    for i in range(firstFrame, lastFrame-1, 1):
        print(SDA.getValueAtTime(i))
        print(threshold)
        if SDA.getValueAtTime(i)>threshold:
            amountOfNodes += 1
            cuts.extend([i])

    cuts.extend([lastFrame])
    positionXStep = (1000 + amountOfNodes // 2) // amountOfNodes
    frameRangePositionX = readerPositionX - positionXStep * (amountOfNodes // 2)
    frameRangePositionY = readerPositionY + 150
    for i in xrange(1,len(cuts)):
        frameRange = app.createNode("net.sf.openfx.FrameRange")
        frameRange.connectInput(0, reader)
        merger.connectInput(i+2, frameRange)
        frameRange.setPosition(frameRangePositionX,frameRangePositionY)
        frameRange.getParam("frameRange").set(cuts[i-1]+1, cuts[i])
        frameRangePositionX = frameRangePositionX + positionXStep


natron.addMenuCommand("PARP/Semiautomatic detection of cuts","generarNodos")



def utilizarEDL():
    file_path          = app1.getFilenameDialog('txt')
    edl_file           = open(file_path, 'r')
    headers            = (edl_file.readline()).split()
    shotNumberPosition = headers.index("Shotnumber")
    firstFramePosition = headers.index("FirstFrame")
    lastFramePosition  = headers.index("LastFrame")
    cuts               = [0]

    for line in edl_file:
        splitted = line.split()
        try:
            gotdata = splitted[lastFramePosition]
        except IndexError:
            gotdata = None

        if (gotdata):
            cuts.append(int(gotdata))

    reader = app.Read1
    viewer = app.Viewer1
    merger = app.createNode("net.sf.openfx.MergePlugin")
    viewer.connectInput(0, merger)
    readerPositionX = reader.getPosition()[0]
    readerPositionY = reader.getPosition()[1]
    merger.setPosition(readerPositionX,readerPositionY+350) # down it goes
    viewer.setPosition(readerPositionX,readerPositionY+600) # down it goes
    amountOfNodes = len(cuts)-1
    positionXStep = (1000 + amountOfNodes // 2) // amountOfNodes
    frameRangePositionX = readerPositionX - positionXStep * (amountOfNodes // 2)
    frameRangePositionY = readerPositionY + 150
    for i in xrange(1,len(cuts)):
        frameRange = app.createNode("net.sf.openfx.FrameRange")
        frameRange.connectInput(0, reader)
        merger.connectInput(i+2, frameRange)
        frameRange.setPosition(frameRangePositionX,frameRangePositionY)
        frameRange.getParam("frameRange").set(cuts[i-1]+1, cuts[i])
        frameRangePositionX = frameRangePositionX + positionXStep

natron.addMenuCommand("PARP/Load EDL","utilizarEDL")