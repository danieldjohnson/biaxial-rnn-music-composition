import cPickle as pickle
import gzip
import numpy
from midi_to_statematrix import *
import os

import multi_training
import model

def gen_adaptive(m,pcs,times,keep_thoughts=False,name="final"):
	xIpt, xOpt = map(lambda x: numpy.array(x, dtype='int8'), multi_training.getPieceSegment(pcs))
	all_outputs = [xOpt[0]]
	if keep_thoughts:
		all_thoughts = []
	m.start_slow_walk(xIpt[0])
	cons = 1
	for time in range(multi_training.batch_len*times):
		resdata = m.slow_walk_fun( cons )
		nnotes = numpy.sum(resdata[-1][:,0])
		if nnotes < 2:
			if cons > 1:
				cons = 1
			cons -= 0.02
		else:
			cons += (1 - cons)*0.3
		all_outputs.append(resdata[-1])
		if keep_thoughts:
			all_thoughts.append(resdata)
	noteStateMatrixToMidi(numpy.array(all_outputs),'output/'+name)
	if keep_thoughts:
		pickle.dump(all_thoughts, open('output/'+name+'.p','wb'))

def fetch_train_thoughts(m,pcs,batches,name="trainthoughts"):
	all_thoughts = []
	for i in range(batches):
		ipt, opt = multi_training.getPieceBatch(pcs)
		thoughts = m.update_thought_fun(ipt,opt)
		all_thoughts.append((ipt,opt,thoughts))
	pickle.dump(all_thoughts, open('output/'+name+'.p','wb'))

if __name__ == '__main__':

	music_path = raw_input("Path to music folder: ")

	pcs = multi_training.loadPieces(music_path)

	print "Pieces loaded, building neural network"
	m = model.Model([300,300],[100,50], dropout=0.5)

	error_interval = 100
	sample_interval = 500
	epochs_per_training = 10000

	composition_name = "composition"
	composition_length = 10

	epochs_trained = 0

	#Create menu
	while True:
		os.system('clear')
		print "TRAINING"
		print "========"
		print ""
		print "Epochs trained:", epochs_trained
		print "Path to music folder:", music_path
		print "1. Error reporting interval:", error_interval
		print "2. Sample creating interval:", sample_interval
		print "3. Amount of epochs to train:", epochs_per_training
		print "4. Train"
		print ""
		print "COMPOSITION"
		print "==========="
		print ""
		print "5. Composition name:", composition_name
		print "6. Composition length:", composition_length
		print "7. Create composition"
		print "8. Exit"
		print ""
		print ""
		command = raw_input("Enter command / variable number:")

		if command == "1":
			error_interval = int(raw_input("Desired error reporting interval: "))

		if command == "2":
			sample_interval = int(raw_input("Desired sample creating interval: "))

		if command == "3":
			epochs_per_training = int(raw_input("Desired epochs per training: "))

		if command == "4":
			print "Training for {} epochs, reporting error every {} epochs and generating a sample every {} epochs".format(epochs_per_training, error_interval, sample_interval)
			multi_training.trainPiece(m, pcs, epochs_per_training, error_interval=error_interval, sample_interval=sample_interval)
			epochs_trained = epochs_trained + epochs_per_training
			pickle.dump( m.learned_config, open( "output/final_learned_config.p", "wb" ) )

		if command == "5":
			composition_name = raw_input("Desired composition name: ")

		if command == "6":
			composition_length = int(raw_input("Desired composition length: "))

		if command == "7":
			print "Creating composition '{}' of length {}".format(composition_name, str(composition_length))
			gen_adaptive(m,pcs,composition_length,name=composition_name)

		if command == "8":
			os._exit(0)


