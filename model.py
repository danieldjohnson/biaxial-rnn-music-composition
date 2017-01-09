import theano, theano.tensor as T
import numpy as np
import theano_lstm

from out_to_in_op import OutputFormToInputFormOp

from theano_lstm import Embedding, LSTM, RNN, StackedCells, Layer, create_optimization_updates, masked_loss, MultiDropout
    
def has_hidden(layer):
    """
    Whether a layer has a trainable
    initial hidden state.
    """
    return hasattr(layer, 'initial_hidden_state')

def matrixify(vector, n):
    # Cast n to int32 if necessary to prevent error on 32 bit systems
    return T.repeat(T.shape_padleft(vector),
                    n if (theano.configdefaults.local_bitwidth() == 64) else T.cast(n,'int32'),
                    axis=0)

def initial_state(layer, dimensions = None):
    """
    Initalizes the recurrence relation with an initial hidden state
    if needed, else replaces with a "None" to tell Theano that
    the network **will** return something, but it does not need
    to send it to the next step of the recurrence
    """
    if dimensions is None:
        return layer.initial_hidden_state if has_hidden(layer) else None
    else:
        return matrixify(layer.initial_hidden_state, dimensions) if has_hidden(layer) else None

def initial_state_with_taps(layer, dimensions = None):
    """Optionally wrap tensor variable into a dict with taps=[-1]"""
    state = initial_state(layer, dimensions)
    if state is not None:
        return dict(initial=state, taps=[-1])
    else:
        return None

class PassthroughLayer(Layer):
    """
    Empty "layer" used to get the final output of the LSTM
    """
    
    def __init__(self):
        self.is_recursive = False
    
    def create_variables(self):
        pass
    
    def activate(self, x):
        return x
    
    @property
    def params(self):
        return []
    
    @params.setter
    def params(self, param_list):
        pass

        
def get_last_layer(result):
    if isinstance(result, list):
        return result[-1]
    else:
        return result

def ensure_list(result):
    if isinstance(result, list):
        return result
    else:
        return [result]
    

class Model(object):
    
    def __init__(self, t_layer_sizes, p_layer_sizes, dropout=0):
        
        self.t_layer_sizes = t_layer_sizes
        self.p_layer_sizes = p_layer_sizes

        # From our architecture definition, size of the notewise input
        self.t_input_size = 80
        
        # time network maps from notewise input size to various hidden sizes
        self.time_model = StackedCells( self.t_input_size, celltype=LSTM, layers = t_layer_sizes)
        self.time_model.layers.append(PassthroughLayer())
        
        # pitch network takes last layer of time model and state of last note, moving upward
        # and eventually ends with a two-element sigmoid layer
        p_input_size = t_layer_sizes[-1] + 2
        self.pitch_model = StackedCells( p_input_size, celltype=LSTM, layers = p_layer_sizes)
        self.pitch_model.layers.append(Layer(p_layer_sizes[-1], 2, activation = T.nnet.sigmoid))
        
        self.dropout = dropout

        self.conservativity = T.fscalar()
        self.srng = T.shared_randomstreams.RandomStreams(np.random.randint(0, 1024))

        self.setup_train()
        self.setup_predict()
        self.setup_slow_walk()

    @property
    def params(self):
        return self.time_model.params + self.pitch_model.params
    
    @params.setter
    def params(self, param_list):
        ntimeparams = len(self.time_model.params)
        self.time_model.params = param_list[:ntimeparams]
        self.pitch_model.params = param_list[ntimeparams:]

    @property
    def learned_config(self):
        return [self.time_model.params, self.pitch_model.params, [l.initial_hidden_state for mod in (self.time_model, self.pitch_model) for l in mod.layers if has_hidden(l)]]

    @learned_config.setter
    def learned_config(self, learned_list):
        self.time_model.params = learned_list[0]
        self.pitch_model.params = learned_list[1]
        for l, val in zip((l for mod in (self.time_model, self.pitch_model) for l in mod.layers if has_hidden(l)), learned_list[2]):
            l.initial_hidden_state.set_value(val.get_value())
    
    def setup_train(self):

        # dimensions: (batch, time, notes, input_data) with input_data as in architecture
        self.input_mat = T.btensor4()
        # dimensions: (batch, time, notes, onOrArtic) with 0:on, 1:artic
        self.output_mat = T.btensor4()
        
        self.epsilon = np.spacing(np.float32(1.0))

        def step_time(in_data, *other):
            other = list(other)
            split = -len(self.t_layer_sizes) if self.dropout else len(other)
            hiddens = other[:split]
            masks = [None] + other[split:] if self.dropout else []
            new_states = self.time_model.forward(in_data, prev_hiddens=hiddens, dropout=masks)
            return new_states
        
        def step_note(in_data, *other):
            other = list(other)
            split = -len(self.p_layer_sizes) if self.dropout else len(other)
            hiddens = other[:split]
            masks = [None] + other[split:] if self.dropout else []
            new_states = self.pitch_model.forward(in_data, prev_hiddens=hiddens, dropout=masks)
            return new_states
        
        # We generate an output for each input, so it doesn't make sense to use the last output as an input.
        # Note that we assume the sentinel start value is already present
        # TEMP CHANGE: NO SENTINEL
        input_slice = self.input_mat[:,0:-1]
        n_batch, n_time, n_note, n_ipn = input_slice.shape
        
        # time_inputs is a matrix (time, batch/note, input_per_note)
        time_inputs = input_slice.transpose((1,0,2,3)).reshape((n_time,n_batch*n_note,n_ipn))
        num_time_parallel = time_inputs.shape[1]
        
        # apply dropout
        if self.dropout > 0:
            time_masks = theano_lstm.MultiDropout( [(num_time_parallel, shape) for shape in self.t_layer_sizes], self.dropout)
        else:
            time_masks = []

        time_outputs_info = [initial_state_with_taps(layer, num_time_parallel) for layer in self.time_model.layers]
        time_result, _ = theano.scan(fn=step_time, sequences=[time_inputs], non_sequences=time_masks, outputs_info=time_outputs_info)
        
        self.time_thoughts = time_result
        
        # Now time_result is a list of matrix [layer](time, batch/note, hidden_states) for each layer but we only care about 
        # the hidden state of the last layer.
        # Transpose to be (note, batch/time, hidden_states)
        last_layer = get_last_layer(time_result)
        n_hidden = last_layer.shape[2]
        time_final = get_last_layer(time_result).reshape((n_time,n_batch,n_note,n_hidden)).transpose((2,1,0,3)).reshape((n_note,n_batch*n_time,n_hidden))
        
        # note_choices_inputs represents the last chosen note. Starts with [0,0], doesn't include last note.
        # In (note, batch/time, 2) format
        # Shape of start is thus (1, N, 2), concatenated with all but last element of output_mat transformed to (x, N, 2)
        start_note_values = T.alloc(np.array(0,dtype=np.int8), 1, time_final.shape[1], 2 )
        correct_choices = self.output_mat[:,1:,0:-1,:].transpose((2,0,1,3)).reshape((n_note-1,n_batch*n_time,2))
        note_choices_inputs = T.concatenate([start_note_values, correct_choices], axis=0)
        
        # Together, this and the output from the last LSTM goes to the new LSTM, but rotated, so that the batches in
        # one direction are the steps in the other, and vice versa.
        note_inputs = T.concatenate( [time_final, note_choices_inputs], axis=2 )
        num_timebatch = note_inputs.shape[1]
        
        # apply dropout
        if self.dropout > 0:
            pitch_masks = theano_lstm.MultiDropout( [(num_timebatch, shape) for shape in self.p_layer_sizes], self.dropout)
        else:
            pitch_masks = []

        note_outputs_info = [initial_state_with_taps(layer, num_timebatch) for layer in self.pitch_model.layers]
        note_result, _ = theano.scan(fn=step_note, sequences=[note_inputs], non_sequences=pitch_masks, outputs_info=note_outputs_info)
        
        self.note_thoughts = note_result
        
        # Now note_result is a list of matrix [layer](note, batch/time, onOrArticProb) for each layer but we only care about 
        # the hidden state of the last layer.
        # Transpose to be (batch, time, note, onOrArticProb)
        note_final = get_last_layer(note_result).reshape((n_note,n_batch,n_time,2)).transpose(1,2,0,3)
        
        # The cost of the entire procedure is the negative log likelihood of the events all happening.
        # For the purposes of training, if the ouputted probability is P, then the likelihood of seeing a 1 is P, and
        # the likelihood of seeing 0 is (1-P). So the likelihood is (1-P)(1-x) + Px = 2Px - P - x + 1
        # Since they are all binary decisions, and are all probabilities given all previous decisions, we can just
        # multiply the likelihoods, or, since we are logging them, add the logs.
        
        # Note that we mask out the articulations for those notes that aren't played, because it doesn't matter
        # whether or not those are articulated.
        # The padright is there because self.output_mat[:,:,:,0] -> 3D matrix with (b,x,y), but we need 3d tensor with 
        # (b,x,y,1) instead
        active_notes = T.shape_padright(self.output_mat[:,1:,:,0])
        mask = T.concatenate([T.ones_like(active_notes),active_notes], axis=3)
        
        loglikelihoods = mask * T.log( 2*note_final*self.output_mat[:,1:] - note_final - self.output_mat[:,1:] + 1 + self.epsilon )
        self.cost = T.neg(T.sum(loglikelihoods))
        
        updates, _, _, _, _ = create_optimization_updates(self.cost, self.params, method="adadelta")
        self.update_fun = theano.function(
            inputs=[self.input_mat, self.output_mat],
            outputs=self.cost,
            updates=updates,
            allow_input_downcast=True)

        self.update_thought_fun = theano.function(
            inputs=[self.input_mat, self.output_mat],
            outputs= ensure_list(self.time_thoughts) + ensure_list(self.note_thoughts) + [self.cost],
            allow_input_downcast=True)
    
    def _predict_step_note(self, in_data_from_time, *states):
        # States is [ *hiddens, last_note_choice ]
        hiddens = list(states[:-1])
        in_data_from_prev = states[-1]
        in_data = T.concatenate([in_data_from_time, in_data_from_prev])

        # correct for dropout
        if self.dropout > 0:
            masks = [1 - self.dropout for layer in self.pitch_model.layers]
            masks[0] = None
        else:
            masks = []

        new_states = self.pitch_model.forward(in_data, prev_hiddens=hiddens, dropout=masks)
        
        # Now new_states is a per-layer set of activations.
        probabilities = get_last_layer(new_states)
        
        # Thus, probabilities is a vector of two probabilities, P(play), and P(artic | play)
        
        shouldPlay = self.srng.uniform() < (probabilities[0] ** self.conservativity)
        shouldArtic = shouldPlay * (self.srng.uniform() < probabilities[1])
        
        chosen = T.cast(T.stack(shouldPlay, shouldArtic), "int8")
        
        return ensure_list(new_states) + [chosen]

    def setup_predict(self):
        # In prediction mode, note steps are contained in the time steps. So the passing gets a little bit hairy.

        self.predict_seed = T.bmatrix()
        self.steps_to_simulate = T.iscalar()

        def step_time(*states):
            # States is [ *hiddens, prev_result, time]
            hiddens = list(states[:-2])
            in_data = states[-2]
            time = states[-1]

            # correct for dropout
            if self.dropout > 0:
                masks = [1 - self.dropout for layer in self.time_model.layers]
                masks[0] = None
            else:
                masks = []

            new_states = self.time_model.forward(in_data, prev_hiddens=hiddens, dropout=masks)
            
            # Now new_states is a list of matrix [layer](notes, hidden_states) for each layer
            time_final = get_last_layer(new_states)
            
            start_note_values = theano.tensor.alloc(np.array(0,dtype=np.int8), 2)
            
            # This gets a little bit complicated. In the training case, we can pass in a combination of the
            # time net's activations with the known choices. But in the prediction case, those choices don't
            # exist yet. So instead of iterating over the combination, we iterate over only the activations,
            # and then combine in the previous outputs in the step. And then since we are passing outputs to
            # previous inputs, we need an additional outputs_info for the initial "previous" output of zero.
            note_outputs_info = ([ initial_state_with_taps(layer) for layer in self.pitch_model.layers ] +
                                 [ dict(initial=start_note_values, taps=[-1]) ])
            
            notes_result, updates = theano.scan(fn=self._predict_step_note, sequences=[time_final], outputs_info=note_outputs_info)
            
            # Now notes_result is a list of matrix [layer/output](notes, onOrArtic)
            output = get_last_layer(notes_result)
            
            next_input = OutputFormToInputFormOp()(output, time + 1) # TODO: Fix time
            #next_input = T.cast(T.alloc(0, 3, 4),'int64')
            
            return (ensure_list(new_states) + [ next_input, time + 1, output ]), updates
        
        # start_sentinel = startSentinel()
        num_notes = self.predict_seed.shape[0]
        
        time_outputs_info = ([ initial_state_with_taps(layer, num_notes) for layer in self.time_model.layers ] +
                             [ dict(initial=self.predict_seed, taps=[-1]),
                               dict(initial=0, taps=[-1]),
                               None ])
            
        time_result, updates = theano.scan( fn=step_time, 
                                            outputs_info=time_outputs_info, 
                                            n_steps=self.steps_to_simulate )
        
        self.predict_thoughts = time_result
        
        self.predicted_output = time_result[-1]
        
        self.predict_fun = theano.function(
            inputs=[self.steps_to_simulate, self.conservativity, self.predict_seed],
            outputs=self.predicted_output,
            updates=updates,
            allow_input_downcast=True)

        self.predict_thought_fun = theano.function(
            inputs=[self.steps_to_simulate, self.conservativity, self.predict_seed],
            outputs=ensure_list(self.predict_thoughts),
            updates=updates,
            allow_input_downcast=True)

    def setup_slow_walk(self):

        self.walk_input = theano.shared(np.ones((2,2), dtype='int8'))
        self.walk_time = theano.shared(np.array(0, dtype='int64'))
        self.walk_hiddens = [theano.shared(np.ones((2,2), dtype=theano.config.floatX)) for layer in self.time_model.layers if has_hidden(layer)]
        
        # correct for dropout
        if self.dropout > 0:
            masks = [1 - self.dropout for layer in self.time_model.layers]
            masks[0] = None
        else:
            masks = []

        new_states = self.time_model.forward(self.walk_input, prev_hiddens=self.walk_hiddens, dropout=masks)

        # Now new_states is a list of matrix [layer](notes, hidden_states) for each layer
        time_final = get_last_layer(new_states)
        
        start_note_values = theano.tensor.alloc(np.array(0,dtype=np.int8), 2)
        note_outputs_info = ([ initial_state_with_taps(layer) for layer in self.pitch_model.layers ] +
                             [ dict(initial=start_note_values, taps=[-1]) ])
        
        notes_result, updates = theano.scan(fn=self._predict_step_note, sequences=[time_final], outputs_info=note_outputs_info)
        
        # Now notes_result is a list of matrix [layer/output](notes, onOrArtic)
        output = get_last_layer(notes_result)
        
        next_input = OutputFormToInputFormOp()(output, self.walk_time + 1) # TODO: Fix time
        #next_input = T.cast(T.alloc(0, 3, 4),'int64')

        slow_walk_results = (new_states[:-1] + notes_result[:-1] + [ next_input, output ])

        updates.update({
                self.walk_time: self.walk_time+1,
                self.walk_input: next_input
            })

        updates.update({hidden:newstate for hidden, newstate, layer in zip(self.walk_hiddens, new_states, self.time_model.layers) if has_hidden(layer)})

        self.slow_walk_fun = theano.function(
            inputs=[self.conservativity],
            outputs=slow_walk_results,
            updates=updates,
            allow_input_downcast=True)

    def start_slow_walk(self, seed):
        seed = np.array(seed)
        num_notes = seed.shape[0]

        self.walk_time.set_value(0)
        self.walk_input.set_value(seed)
        for layer, hidden in zip((l for l in self.time_model.layers if has_hidden(l)),self.walk_hiddens):
            hidden.set_value(np.repeat(np.reshape(layer.initial_hidden_state.get_value(), (1,-1)), num_notes, axis=0))
        














