import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def actToColor(memcell, activation):
	return [0, sigmoid(activation), sigmoid(memcell)]

def internalMatrixToImgArray(inmat):
	return np.array(
		[[actToColor(m,a) for m,a in zip(row[:len(row)/2],row[len(row)/2:])]
			for row in inmat])

def probAndSuccessToImgArray(prob, succ, idx):
	return np.array([[[pr[idx]]*3,[sr[idx],0,0]] for pr, sr in zip(prob, succ)])

def thoughtsToImageArray(thoughts):
	
	spacer = np.zeros((thoughts[0].shape[0], 5, 3))

	sequence = [
			spacer,
			probAndSuccessToImgArray(thoughts[4],thoughts[6], 0),
			spacer,
			probAndSuccessToImgArray(thoughts[4],thoughts[6], 1)
	]

	for thought in thoughts[:-3]:
		sequence = [ spacer, internalMatrixToImgArray(thought) ] + sequence

	return (np.concatenate(sequence, axis=1 )*255).astype('uint8')

def pastColor(prob, succ):
	return [prob[0], succ[0], succ[1]*succ[0]]

def drawPast(probs, succs):
	return np.array([
		[
			pastColor(probs[time][note_idx], succs[time][note_idx])
			for time in range(len(probs))
		]
		for note_idx in range(len(probs[0]))
	])

def thoughtsAndPastToStackedArray(thoughts, probs, succs, len_past):
	
	vert_spacer = np.zeros((thoughts[0].shape[0], 5, 3))

	past_out = drawPast(probs, succs)

	if len(probs) < len_past:
		past_out = np.pad(past_out, ((0,0),(len_past-len(probs),0),(0,0)), mode='constant')

	def add_cur(ipt):
		return np.concatenate((
			ipt, 
			vert_spacer, 
			probAndSuccessToImgArray(thoughts[-3],thoughts[-1], 0),
			vert_spacer,
			probAndSuccessToImgArray(thoughts[-3],thoughts[-1], 1)), axis=1)

	horiz_spacer = np.zeros((5, 1, 3))

	rows = [add_cur(past_out[-len_past:])]

	for thought in thoughts[:-3]:
		rows += [ horiz_spacer, add_cur(internalMatrixToImgArray(thought)) ]

	maxlen = max([x.shape[1] for x in rows])
	rows = [np.pad(row, ((0,0),(maxlen-row.shape[1],0),(0,0)), mode='constant') for row in rows]

	return (np.concatenate(rows, axis=0 )*255).astype('uint8')