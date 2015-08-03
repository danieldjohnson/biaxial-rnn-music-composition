import cPickle as pickle
import gzip
import numpy
from midi_to_statematrix import *

rel_modules = []

import multi_training
rel_modules.append(multi_training)
import model
rel_modules.append(model)

def refresh():
	for mod in rel_modules:
		reload(mod)

pcs = multi_training.loadPieces("music")
# pickle.dump( pcs, gzip.GzipFile( "traindata.p.zip", "wb" ) )
# pcs = pickle.load(gzip.GzipFile( "traindata.p.zip", "r"))

m = model.Model([300,300],[100,50], dropout=0.5)

multi_training.trainPiece(m, pcs, 10000)

pickle.dump( m.learned_config, open( "output/final_learned_config.p", "wb" ) )

def gen_adaptive(times,keep_thoughts=False,name="final"):
	xIpt, xOpt = map(lambda x: numpy.array(x, dtype='int8'), multi_training.getPieceSegment(pcs))
	all_outputs = [xOpt[0]]
	if keep_thoughts:
		all_thoughts = []
	m.start_slow_walk(xIpt[0])
	cons = 1
	for time in range(multi_training.batch_len*times):
		resdata = m.slow_walk_fun( cons )
		nnotes = np.sum(resdata[-1][:,0])
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

def fetch_train_thoughts(batches,name="trainthoughts"):
	all_thoughts = []
	for i in range(batches):
		ipt, opt = multi_training.getPieceBatch(pcs)
		thoughts = m.update_thought_fun(ipt,opt)
		all_thoughts.append((ipt,opt,thoughts))
	pickle.dump(all_thoughts, open('output/'+name+'.p','wb'))


