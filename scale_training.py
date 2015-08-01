import numpy
import random
from midi_to_statematrix import *
from data import *

def generateScaleNotes():
    scaletypes = [
        [0,2,4,5,7,9,11,12],
        [0,2,3,5,7,8,10,12],
        [0,2,3,5,7,8,11,12],
    ]
    start = random.randint(lowerBound, upperBound-13)
    return [x+start for x in random.choice(scaletypes)]

def generateScaleSequence():
    notes = generateScaleNotes()
    pattern = [
        [(0,True)],
        [(0,False)],
        [(1,True)],
        [(1,False)],
        [(2,True)],
        [(2,False)],
        [(3,True)],
        [(3,False)],
        [(4,True)],
        [(4,False)],
        [(5,True)],
        [(5,False)],
        [(6,True)],
        [(6,False)],
        [(7,True)],
        [(7,False)],

        [(7,True)],
        [(7,False)],
        [(6,True)],
        [(6,False)],
        [(5,True)],
        [(5,False)],
        [(4,True)],
        [(4,False)],
        [(3,True)],
        [(3,False)],
        [(2,True)],
        [(2,False)],
        [(1,True)],
        [(1,False)],
        [(0,True)],
        [(0,False)],

        [(0,True)],
        [(2,True)],
        [(4,True)],
        [(7,True)],
        [(4,True)],
        [(2,True)],
        [(0,True)],
        [(0,False)],

        [],
        [],
        [(0,True),(2,True),(4,True)],
        [],
        [(0,True),(2,True),(4,True)],
        [(0,False),(2,False),(4,False)],
    ]
    sequence = [[(notes[note], artic) for note, artic in step] for step in pattern]
    return sequence   

def sequenceToNoteStateMatrix(seq):
    statematrix = []
    span = upperBound-lowerBound
    for noteset in seq:
        state = [[0,0] for x in range(span)]
        for note, artic in noteset:
            state[note-lowerBound] = [1, int(artic)]
        statematrix.append(state)
    return statematrix

def trainScales(model,epochs):
    def getOpt():
        return sequenceToNoteStateMatrix(generateScaleSequence())
    def getIpt(opt):
        return noteStateMatrixToInputForm(opt)
    def getBatchIO(num):
        outs = [getOpt() for _ in range(num)]
        ins = [getIpt(out) for out in outs]
        return numpy.array(ins), numpy.array(outs)

    for i in range(epochs):
        error = model.update_fun(*getBatchIO(10))
        if i % 100 == 0:
            print "epoch {}, error={}".format(i,error)
        if i % 500 == 0 or (i % 100 == 0 and i < 1000):
            xOpt = getOpt()
            xIpt = getIpt(xOpt)
            noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), model.predict_fun(40, 1, xIpt[0])), axis=0),i)
        i += 1