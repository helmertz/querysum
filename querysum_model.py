from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, GRUCell
from tensorflow.python.ops import rnn
from tensorflow.python.training.adam import AdamOptimizer


class QuerySumModel(object):
    def __init__(self, mode, word_dict, word_embedding_dim, vocabulary, initial_vocabulary_embeddings,
                 target_vocabulary_size):
        self.word_dict = word_dict
        self.word_embedding_dim = word_embedding_dim
        self.summary_vocabulary = vocabulary
        self.target_vocabulary_size = min(len(vocabulary.words), target_vocabulary_size)
        self.embeddings = tf.Variable(initial_vocabulary_embeddings, name='embeddings')

        self.documents_placeholder = tf.placeholder(tf.int32, shape=[None, None])
        self.document_lengths_placeholder = tf.placeholder(tf.int32, shape=[None])
        self.queries_placeholder = tf.placeholder(tf.int32, shape=[None, None])
        self.query_lengths_placeholder = tf.placeholder(tf.int32, shape=[None])
        self.references_placeholder = tf.placeholder(tf.int32, shape=[None, None])
        self.reference_lengths_placeholder = tf.placeholder(tf.int32, shape=[None])
        self.pointer_reference_placeholder = tf.placeholder(tf.int32, shape=[None, None])
        self.pointer_switch_placeholder = tf.placeholder(tf.int32, shape=[None, None])
        self.reference_lengths_placeholder = tf.placeholder(tf.int32, shape=[None])

        self.epoch = tf.Variable(0, name='epoch', trainable=False)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.best_validation_loss = tf.Variable(np.inf, name='best_validation_loss', trainable=False)
        self.new_best_validation = tf.placeholder(tf.float32, shape=[])
        self.best_validation_assign = self.best_validation_loss.assign(self.new_best_validation)

        self.increment_epoch_op = tf.assign(self.epoch, self.epoch + 1)

        self.batch_size = tf.shape(self.documents_placeholder)[0]

        self.dropout_enabled = False

        self.encoder_cell_state_size = 256
        self.encoder_output_size = 2 * self.encoder_cell_state_size
        self.decoder_cell_state_size = self.encoder_output_size

        self.decoder_vocab_hidden_size = 256

        self.attention_hidden_output_size = 256
        # Size is that of decoder state + encoder hidden state + query reader state
        self.attention_hidden_input_size = (self.decoder_cell_state_size + self.encoder_output_size +
                                            self.encoder_cell_state_size)

        self.beam_width_placeholder = tf.placeholder(tf.int32, shape=[])
        self.decode_last_output_placeholder = tf.placeholder(tf.int32, shape=[None])

        self.initial_decoder_state_placeholder = tf.placeholder(
            tf.float32,
            shape=[None, self.decoder_cell_state_size])

        self.pre_computed_encoder_states_placeholder = tf.placeholder(
            tf.float32,
            shape=[None, None, self.encoder_output_size])

        self.pre_computed_query_state_placeholder = tf.placeholder(
            tf.float32,
            shape=[None, self.encoder_cell_state_size])

        self.query_attention_partial_score_placeholder = tf.placeholder(
            tf.float32,
            shape=[None, self.attention_hidden_output_size])

        self.encoder_state_attention_partial_scores_placeholder = tf.placeholder(
            tf.float32,
            shape=[None, None, self.attention_hidden_output_size])

        self.mode = mode
        self._build_graph(mode=mode)

    def _build_graph(self, mode):
        self._add_encoders()
        self._add_decoder(mode)
        if mode == 'train':
            self._add_optimizer()

    def _add_encoders(self):
        with tf.variable_scope('query_encoder'):
            query_encoder_cell = GRUCell(self.encoder_cell_state_size)
            if self.dropout_enabled and self.mode != 'decode':
                query_encoder_cell = DropoutWrapper(cell=query_encoder_cell, output_keep_prob=0.8)

            query_embeddings = tf.nn.embedding_lookup(self.embeddings, self.queries_placeholder)

            query_encoder_outputs, _ = rnn.dynamic_rnn(query_encoder_cell, query_embeddings,
                                                       sequence_length=self.query_lengths_placeholder,
                                                       swap_memory=True, dtype=tf.float32)
            self.query_last = query_encoder_outputs[:, -1, :]

        with tf.variable_scope('encoder'):
            fw_cell = GRUCell(self.encoder_cell_state_size)
            bw_cell = GRUCell(self.encoder_cell_state_size)

            if self.dropout_enabled and self.mode != 'decode':
                fw_cell = DropoutWrapper(cell=fw_cell, output_keep_prob=0.8)
                bw_cell = DropoutWrapper(cell=bw_cell, output_keep_prob=0.8)

            embeddings = tf.nn.embedding_lookup(self.embeddings, self.documents_placeholder)

            (encoder_outputs_fw, encoder_outputs_bw), _ = rnn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell,
                embeddings,
                sequence_length=self.document_lengths_placeholder,
                swap_memory=True,
                dtype=tf.float32)

            self.encoder_outputs = tf.concat([encoder_outputs_fw, encoder_outputs_bw], 2)

            self.final_encoder_state = self.encoder_outputs[:, -1, :]

    def _add_decoder(self, mode):

        with tf.variable_scope('decoder') as scope:
            decoder_cell = GRUCell(self.decoder_cell_state_size)
            if self.dropout_enabled and self.mode != 'decode':
                decoder_cell = DropoutWrapper(cell=decoder_cell, output_keep_prob=0.8)

            self.vocabulary_project_w_1 = tf.get_variable(
                name='vocabulary_project_w_1',
                shape=[decoder_cell.output_size + self.encoder_output_size, self.decoder_vocab_hidden_size])

            self.vocabulary_project_w_2 = tf.get_variable(
                name='vocabulary_project_w_2',
                shape=[self.decoder_vocab_hidden_size, self.target_vocabulary_size])

            self.vocabulary_project_b_1 = tf.get_variable(
                name='vocabulary_project_b_1',
                initializer=tf.zeros_initializer(),
                shape=[self.decoder_vocab_hidden_size])

            self.vocabulary_project_b_2 = tf.get_variable(
                name='vocabulary_project_b_2',
                initializer=tf.zeros_initializer(),
                shape=[self.target_vocabulary_size])

            self.pointer_probability_project_w = tf.get_variable(
                name='pointer_probability_project_w',
                shape=[
                    self.encoder_output_size +
                    self.decoder_cell_state_size +
                    self.word_embedding_dim
                    , 1])

            self.pointer_probability_project_b = tf.get_variable(
                name='pointer_probability_project_b',
                initializer=tf.zeros_initializer(),
                shape=[1])

            self.attention_w = tf.get_variable(
                name='attention_w',
                shape=[self.decoder_cell_state_size, self.attention_hidden_output_size],
                dtype=tf.float32)

            self.attention_w_e = tf.get_variable(
                name='attention_w_e',
                shape=[self.word_embedding_dim,
                       self.attention_hidden_output_size],
                dtype=tf.float32)

            self.attention_w_q = tf.get_variable(
                name='attention_w_q',
                shape=[self.encoder_cell_state_size,
                       self.attention_hidden_output_size],
                dtype=tf.float32)

            self.attention_w_d = tf.get_variable(
                name='attention_w_d',
                shape=[self.encoder_output_size, self.attention_hidden_output_size],
                dtype=tf.float32)

            self.attention_v = tf.get_variable(
                name='attention_v',
                shape=[self.attention_hidden_output_size],
                dtype=tf.float32)

            self.attention_b = tf.get_variable(
                name='attention_b',
                initializer=tf.zeros_initializer(),
                shape=[self.attention_hidden_output_size],
                dtype=tf.float32)

            self._precompute_partial_attention_scores()

            if mode == 'decode':
                embedding = tf.nn.embedding_lookup(self.embeddings, self.decode_last_output_placeholder)
                (decoder_outputs, self.one_step_decoder_state, context_vectors, attention_logits,
                 pointer_probabilities) = self._rnn_one_step_attention_decoder(decoder_cell, embedding,
                                                                               self.initial_decoder_state_placeholder)
            else:
                if mode == 'train':
                    (train_decoder_outputs, train_context_vectors, train_attention_logits,
                     train_pointer_probabilities) = self._rnn_attention_decoder(decoder_cell, training_wheels=True)
                    scope.reuse_variables()

                    self.train_attention_argmax = tf.cast(tf.argmax(train_attention_logits, 1), dtype=tf.int32)
                    self.train_pointer_enabled = tf.cast(tf.round(train_pointer_probabilities), tf.int32)
                (decoder_outputs, context_vectors, attention_logits,
                 pointer_probabilities) = self._rnn_attention_decoder(decoder_cell, training_wheels=False)

        self.attention_argmax = tf.cast(tf.argmax(attention_logits, 1), dtype=tf.int32)
        self.attention_softmax = tf.nn.softmax(attention_logits)
        self.pointer_enabled = tf.cast(tf.round(pointer_probabilities), tf.int32)

        if mode == 'decode':
            self.top_k_vocabulary_argmax, self.top_k_probabilities = self._extract_top_k_argmax(
                self.beam_width_placeholder, decoder_outputs, context_vectors)
        else:
            if mode == 'train':
                self.train_vocabulary_argmax, self.main_train_loss = self._compute_argmax_and_loss(
                    train_decoder_outputs, train_context_vectors, train_attention_logits, train_pointer_probabilities)
            self.vocabulary_argmax, self.main_loss = self._compute_argmax_and_loss(
                decoder_outputs, context_vectors, attention_logits, pointer_probabilities)

    def _rnn_attention_decoder(self, decoder_cell, training_wheels):
        loop_fn = self._custom_rnn_loop_fn(decoder_cell.output_size, training_wheels=training_wheels)
        decoder_outputs, _, (context_vectors_array, attention_logits_array, pointer_probability_array) = \
            tf.nn.raw_rnn(decoder_cell,
                          loop_fn,
                          swap_memory=True)

        decoder_outputs = decoder_outputs.stack()
        decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])

        attention_logits = attention_logits_array.gather(tf.range(0, attention_logits_array.size() - 1))
        attention_logits = tf.transpose(attention_logits, [1, 0, 2])

        context_vectors = context_vectors_array.gather(tf.range(0, context_vectors_array.size() - 1))
        context_vectors = tf.transpose(context_vectors, [1, 0, 2])

        pointer_probabilities = pointer_probability_array.gather(tf.range(0, pointer_probability_array.size() - 1))
        pointer_probabilities = tf.transpose(pointer_probabilities, [1, 0])

        return decoder_outputs, context_vectors, attention_logits, pointer_probabilities

    def _custom_rnn_loop_fn(self, cell_size, training_wheels):
        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:  # time == 0
                context_vectors_array = tf.TensorArray(tf.float32, size=tf.shape(self.references_placeholder)[1] + 1)
                attention_logits_array = tf.TensorArray(tf.float32, size=tf.shape(self.references_placeholder)[1] + 1)
                pointer_probability_array = tf.TensorArray(tf.float32,
                                                           size=tf.shape(self.references_placeholder)[1] + 1)
                next_cell_state = self.final_encoder_state
                go_id = self.summary_vocabulary.word_to_id('<GO>')
                last_output_embedding = tf.nn.embedding_lookup(self.embeddings, tf.tile([go_id], [self.batch_size]))
            else:
                context_vectors_array, attention_logits_array, pointer_probability_array = loop_state
                next_cell_state = cell_state

                if training_wheels:
                    voc_indices = self.references_placeholder[:, time - 1]
                    pointer_indices = self.pointer_reference_placeholder[:, time - 1]
                    pointer_switch = tf.cast(self.pointer_switch_placeholder[:, time - 1], tf.bool)

                    batch_range = tf.range(self.batch_size)
                    pointer_indexer = tf.stack([batch_range, pointer_indices], axis=1)
                    attention_vocabulary_indices = tf.gather_nd(self.documents_placeholder, pointer_indexer)

                    mixed_indices = tf.where(pointer_switch, attention_vocabulary_indices, voc_indices)
                    last_output_embedding = tf.nn.embedding_lookup(self.embeddings, mixed_indices)
                else:
                    last_output_embedding = self._extract_argmax_and_embed(cell_output, cell_size,
                                                                           tf.shape(self.documents_placeholder)[0])
            context_vector, attention_logits = self._attention(next_cell_state, last_output_embedding)
            pointer_probabilities = self._pointer_probabilities(context_vector, next_cell_state, last_output_embedding)

            context_vectors_array = context_vectors_array.write(time, context_vector)
            attention_logits_array = attention_logits_array.write(time, attention_logits)
            pointer_probability_array = pointer_probability_array.write(time, pointer_probabilities)

            next_input = tf.concat([last_output_embedding, context_vector, self.query_last], axis=1)
            elements_finished = (time >= self.reference_lengths_placeholder)

            emit_output = cell_output
            next_loop_state = (context_vectors_array, attention_logits_array, pointer_probability_array)
            return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

        return loop_fn

    def _precompute_partial_attention_scores(self):
        encoder_outputs_flat = tf.reshape(self.encoder_outputs, shape=[-1, self.encoder_output_size])
        self.encoder_state_attention_partial_scores = tf.matmul(encoder_outputs_flat, self.attention_w_d)
        self.encoder_state_attention_partial_scores = tf.reshape(self.encoder_state_attention_partial_scores,
                                                                 shape=[self.batch_size, -1,
                                                                        self.attention_hidden_output_size])
        self.encoder_state_attention_partial_scores = tf.transpose(self.encoder_state_attention_partial_scores,
                                                                   [1, 0, 2])

        self.query_attention_partial_score = tf.matmul(self.query_last, self.attention_w_q)

    def _score(self, prev_decoder_state, prev_embedding):
        # Returns scores in a tensor of shape [batch_size, input_sequence_length]

        if self.mode == 'decode':
            query_part = self.query_attention_partial_score_placeholder
            encoder_part = self.encoder_state_attention_partial_scores_placeholder
        else:
            query_part = self.query_attention_partial_score
            encoder_part = self.encoder_state_attention_partial_scores

        embedding_part = tf.matmul(prev_embedding, self.attention_w_e)

        output = tf.matmul(prev_decoder_state,
                           self.attention_w) + embedding_part + query_part + encoder_part + self.attention_b
        output = tf.tanh(output)
        output = tf.reduce_sum(self.attention_v * output, axis=2)
        output = tf.transpose(output, [1, 0])

        # Handle input document padding by giving a large penalty, eliminating it from the weighted average
        padding_penalty = -1e20 * tf.to_float(1 - tf.sign(self.documents_placeholder))
        masked = output + padding_penalty

        return masked

    def _attention(self, prev_decoder_state, prev_embedding):
        with tf.variable_scope('attention') as scope:
            # e = score of shape [batch_size, output_seq_length, input_seq_length], e_{ij} = score(s_{i-1}, h_j)
            # e_i = score of shape [batch_size, input_seq_length], e_ij = score(prev_decoder_state, h_j)
            e_i = self._score(prev_decoder_state, prev_embedding)

            # alpha_i = softmax(e_i) of shape [batch_size, input_seq_length]
            alpha_i = tf.nn.softmax(e_i)

            resized_alpha_i = tf.reshape(tf.tile(alpha_i, [1, self.encoder_output_size]),
                                         [self.batch_size, -1, self.encoder_output_size])

            if self.mode == 'decode':
                c_i = tf.reduce_sum(tf.multiply(resized_alpha_i, self.pre_computed_encoder_states_placeholder), axis=1)
            else:
                c_i = tf.reduce_sum(tf.multiply(resized_alpha_i, self.encoder_outputs), axis=1)
            return c_i, e_i

    def _pointer_probabilities(self, attention, cell_state, last_output_embedding):
        combined_input = tf.concat([attention, cell_state, last_output_embedding], axis=1)
        result = tf.sigmoid(tf.matmul(combined_input, self.pointer_probability_project_w) +
                            self.pointer_probability_project_b)
        # Remove extra dimension of size 1
        result = tf.reshape(result, shape=[self.batch_size])
        return result

    def _compute_argmax_and_loss(self, decoder_outputs, context_vectors, attention_logits, pointer_probabilities):
        # Projection onto vocabulary is based on
        # http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/

        vocabulary_project_input = tf.concat([decoder_outputs, context_vectors], axis=2)

        # Flatten output over batch dimension
        vocabulary_project_input_flat = tf.reshape(vocabulary_project_input,
                                                   [-1, self.decoder_cell_state_size + self.encoder_output_size])
        vocabulary_hidden_flat = tf.matmul(vocabulary_project_input_flat,
                                           self.vocabulary_project_w_1) + self.vocabulary_project_b_1

        logits_flat = tf.matmul(vocabulary_hidden_flat, self.vocabulary_project_w_2) + self.vocabulary_project_b_2

        max_decoder_length = tf.shape(decoder_outputs)[1]

        # Reshape back to [batch_size, max_decoder_length, vocabulary_size]
        logits = tf.reshape(logits_flat, [-1, max_decoder_length, self.target_vocabulary_size])

        vocabulary_argmax = tf.argmax(logits, 2)

        references_placeholder_flat = tf.reshape(self.references_placeholder, [-1, 1])

        # Calculate the losses
        losses_flat = tf.nn.sampled_softmax_loss(
            weights=tf.transpose(self.vocabulary_project_w_2),
            biases=self.vocabulary_project_b_2,
            labels=references_placeholder_flat,
            inputs=vocabulary_hidden_flat,
            num_sampled=512,
            num_classes=self.target_vocabulary_size
        )
        vocabulary_loss = tf.reshape(losses_flat, [-1, max_decoder_length])

        # Previous loss function for full softmax
        # vocabulary_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
        #                                                                 labels=self.references_placeholder)

        pointer_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=attention_logits,
                                                                      labels=self.pointer_reference_placeholder)

        float_pointer_switch_reference = tf.to_float(self.pointer_switch_placeholder)

        pointer_probability_loss = (float_pointer_switch_reference * -tf.log(pointer_probabilities + 1e-9) +
                                    (1. - float_pointer_switch_reference) * -tf.log(1. - pointer_probabilities + 1e-9))

        # Mask out padding from loss computation
        length_mask = tf.sign(tf.to_float(self.references_placeholder))

        masked_losses = length_mask * (
            pointer_probability_loss +
            (1. - float_pointer_switch_reference) * vocabulary_loss +
            float_pointer_switch_reference * pointer_loss
        )

        float_lengths = tf.to_float(self.reference_lengths_placeholder)

        # Calculate mean loss
        mean_loss_by_example = tf.reduce_sum(masked_losses, axis=1) / float_lengths

        mean_loss = tf.reduce_mean(mean_loss_by_example)

        return vocabulary_argmax, mean_loss

    def _extract_argmax_and_embed(self, cell_output, cell_size, batch_size):
        # Flatten output over batch dimension
        rnn_outputs_flat = tf.reshape(cell_output, [-1, cell_size])

        # Running without training wheels is currently not supported
        # TODO: Fix or remove
        logits_flat = tf.zeros([batch_size, self.target_vocabulary_size])
        # logits_flat = tf.matmul(rnn_outputs_flat, self.vocabulary_project_w) + self.vocabulary_project_b

        # Reshape back to [batch_size, vocabulary_size]
        logits = tf.reshape(logits_flat, [-1, self.target_vocabulary_size])
        vocabulary_argmax = tf.argmax(logits, 1)

        return tf.nn.embedding_lookup(self.embeddings, vocabulary_argmax)

    def _add_optimizer(self):
        self.optimizer = AdamOptimizer()

        self.final_train_loss = self.main_train_loss

        with tf.variable_scope('l2_regularization'):
            # Find variables to regularize by iterating over all variables and checking if in set. Haven't found way to
            # directly get variables by absolute path.
            l2_regularized_names = {
                'encoder/bidirectional_rnn/fw/gru_cell/gates/weights:0'
                # If used, add additional complete variables names
            }
            l2_regularized = [variable for variable in tf.trainable_variables() if
                              variable.name in l2_regularized_names]
            l2_loss = 0.001 * tf.add_n([tf.nn.l2_loss(variable) for variable in l2_regularized])

            # self.train_loss += l2_loss

        gradients = self.optimizer.compute_gradients(self.final_train_loss)

        with tf.variable_scope('gradient_clipping'):
            def clip_gradient(gradient, variable):
                # Only clip normal tensors, IndexedSlices gives warning otherwise
                if isinstance(gradient, tf.Tensor):
                    gradient = tf.clip_by_norm(gradient, 10)
                return gradient, variable

            gradients = [clip_gradient(gradient, variable) for gradient, variable in gradients]
        self.minimize_operation = self.optimizer.apply_gradients(gradients, global_step=self.global_step)

    def _rnn_one_step_attention_decoder(self, decoder_cell, initial_input_word_embedding, initial_cell_state):
        loop_fn = self._custom_one_step_rnn_loop_fn(initial_input_word_embedding, initial_cell_state)
        decoder_outputs, final_state, (context_vector, attention_logits, pointer_probabilities) = tf.nn.raw_rnn(
            decoder_cell, loop_fn)
        decoder_outputs = decoder_outputs.stack()
        decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])
        return decoder_outputs, final_state, context_vector, attention_logits, pointer_probabilities

    def _custom_one_step_rnn_loop_fn(self, initial_input_word_embedding, initial_cell_state):
        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:  # time == 0
                next_cell_state = initial_cell_state
                context_vector, attention_logits = self._attention(next_cell_state, initial_input_word_embedding)
                pointer_probabilities = self._pointer_probabilities(context_vector, next_cell_state,
                                                                    initial_input_word_embedding)
                next_input = tf.concat(
                    [initial_input_word_embedding, context_vector, self.pre_computed_query_state_placeholder], axis=1)
                next_loop_state = (context_vector, attention_logits, pointer_probabilities)
            else:
                next_cell_state = cell_state
                next_input = tf.zeros(shape=[self.batch_size,
                                             self.word_embedding_dim +
                                             self.encoder_output_size +
                                             self.encoder_cell_state_size
                                             ])
                next_loop_state = loop_state

            elements_finished = cell_output is not None

            emit_output = cell_output
            return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

        return loop_fn

    def _extract_top_k_argmax(self, k, cell_output, context_vectors):
        cell_output_flat = tf.reshape(cell_output, [-1, self.decoder_cell_state_size])
        vocabulary_project_input = tf.concat([cell_output_flat, context_vectors], axis=1)

        vocabulary_hidden = tf.matmul(vocabulary_project_input,
                                      self.vocabulary_project_w_1) + self.vocabulary_project_b_1

        logits = tf.matmul(vocabulary_hidden, self.vocabulary_project_w_2) + self.vocabulary_project_b_2

        top_k_probabilities, vocabulary_argmax = tf.nn.top_k(tf.nn.softmax(logits), k)

        return vocabulary_argmax, top_k_probabilities
