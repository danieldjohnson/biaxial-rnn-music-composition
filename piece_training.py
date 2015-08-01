import numpy
from midi_to_statematrix import *
from data import *

batch_width = 5 # number of sequences in a batch
batch_len = 16*4 # length of each sequence
division_len = 16 # interval between possible start locations


piece_output = midiToNoteStateMatrix("bach.mid")
piece_input = noteStateMatrixToInputForm(piece_output)

print "Piece has length {} -> seqs are 1/{}".format(len(piece_output), len(piece_output)/batch_len)

def getPieceSegment():
    start = random.randrange(0,len(piece_output)-batch_len,division_len)
    return piece_input[start:start+batch_len], piece_output[start:start+batch_len]

def getPieceBatch():
    i,o = zip(*[getSegment() for _ in range(batch_width)])
    return numpy.array(i), numpy.array(o)

def trainPiece(model,epochs):
	for i in range(epochs)
	    error = model.update_fun(*getPieceBatch())
	    if i % 100 == 0:
	        print "epoch {}, error={}".format(i,error)
	    if i % 500 == 0 or (i % 100 == 0 and i < 1000):
	        xIpt, xOpt = map(numpy.array, getPieceSegment())
	        noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), model.predict_fun(batch_len, 1, xIpt[0])), axis=0),i)