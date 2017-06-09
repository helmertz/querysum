from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import io
import math
import os
from collections import defaultdict
from os import path
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from batcher import Batcher
from querysum_model import QuerySumModel
from vocabulary import Vocabulary


class RunContext:
    pass


class ProcessedBatch:
    def __init__(self, documents, document_lengths, queries, query_lengths, references=None, reference_lengths=None,
                 pointer_reference=None, pointer_reference_lengths=None, pointer_switch=None,
                 pointer_switch_lengths=None):
        self.documents = documents
        self.document_lengths = document_lengths
        self.queries = queries
        self.query_lengths = query_lengths
        self.references = references
        self.reference_lengths = reference_lengths
        self.pointer_reference = pointer_reference
        self.pointer_reference_lengths = pointer_reference_lengths
        self.pointer_switch = pointer_switch
        self.pointer_switch_lengths = pointer_switch_lengths


class Hypothesis:
    def __init__(self, probability, voc_argmax, attention_argmax, next_decoder_state, attention_softmax,
                 pointer_enabled):
        self.probability = probability
        self.voc_argmax = voc_argmax
        self.attention_argmax = attention_argmax
        self.next_decoder_state = next_decoder_state
        self.attention_softmax = attention_softmax
        self.pointer_enabled = pointer_enabled


def main(unused):
    parser = argparse.ArgumentParser()
    parser.add_argument('embedding_path')
    parser.add_argument('vocabulary_dir')
    parser.add_argument('--training_dir')
    parser.add_argument('--validation_dir')
    parser.add_argument('--decode_dir')
    parser.add_argument('--decode_out_dir')
    parser.add_argument('--mode', choices=['train', 'validate', 'decode'], default='train')
    parser.add_argument('--logdir')
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--validation_interval', type=int, default=20000)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--max_output_length', type=int, default=32)
    parser.add_argument('--target_vocabulary_size', type=int, default=20000)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--allow_gpu_growth', action='store_true')
    parser.add_argument('--collect_run_metadata', action='store_true')
    parser.add_argument('--log_weight_images', action='store_true')
    options = parser.parse_args()

    if options.mode == 'decode':
        # Batching not supported in decoding
        options.batch_size = 1

    embedding_words, word_dict, word_embedding_dim = load_word_embeddings(options.embedding_path)

    vocabulary = Vocabulary()
    summary_vocabulary_path = path.join(options.vocabulary_dir, 'summary_vocabulary.txt')
    vocabulary.add_from_file(summary_vocabulary_path, options.target_vocabulary_size - len(vocabulary.words))

    document_vocabulary_path = path.join(options.vocabulary_dir, 'document_vocabulary.txt')

    # Add the most common words from vocabulary
    vocabulary.add_from_file(document_vocabulary_path, 150000)

    # Add additional common words from loaded embeddings
    vocabulary.add_words(embedding_words[:100000])

    run(options, word_dict, word_embedding_dim, vocabulary)


def load_word_embeddings(path_):
    words = []
    word_dict = {}
    with io.open(path_, encoding='utf-8') as file:
        for line in file:
            elements = line.split()
            word = elements[0]
            embedding = np.array(elements[1:], dtype=np.float32)
            words.append(word)
            word_dict[word] = embedding

    # Get dimension from arbitrary element
    n_dim = len(word_dict[next(iter(word_dict))])
    return words, word_dict, n_dim


def create_embedding_matrix(vocabulary, word_dict, word_embedding_dim):
    embeddings_matrix = np.asarray([arr for arr in word_dict.values()], dtype=np.float32)
    mean = np.mean(embeddings_matrix, axis=0)
    sd = embeddings_matrix.std(axis=0)

    embeddings = np.random.normal(mean, sd, [len(vocabulary.words), word_embedding_dim]).astype(np.float32)

    for index, word in enumerate(vocabulary.words):
        embedding = word_dict.get(word)
        if embedding is not None:
            embeddings[index] = embedding
    return embeddings


def run(options, word_dict, word_embedding_dim, vocabulary):
    embeddings = create_embedding_matrix(vocabulary, word_dict, word_embedding_dim)

    model = QuerySumModel(options.mode, word_dict, word_embedding_dim, vocabulary, embeddings,
                          options.target_vocabulary_size)

    training_batcher = None
    validation_batcher = None
    decode_batcher = None
    if options.mode == 'decode':
        decode_batcher = Batcher(options.decode_dir, options.batch_size, options.synthetic, reference_looping=False)
    else:
        if options.mode == 'train':
            training_batcher = Batcher(options.training_dir, options.batch_size, options.synthetic)
        validation_batcher = Batcher(options.validation_dir, options.batch_size, options.synthetic, max_count=3000)

    training_summary_op = None
    if options.mode == 'train':
        tf.summary.scalar('main_train_loss', model.main_train_loss, collections=['training'])
        tf.summary.scalar('final_train_loss', model.final_train_loss, collections=['training'])
        training_summary_op = tf.summary.merge_all('training')

    weight_images_summary_op = None
    if options.log_weight_images:
        for variable in tf.trainable_variables():
            name = variable.name
            shape = variable.get_shape()

            height = shape.dims[0].value
            if shape.ndims == 1:
                width = 1
            elif shape.ndims == 2:
                width = shape.dims[1].value
            else:
                continue

            if width < height:
                variable = tf.transpose(variable)
                width, height = height, width

            tensor_as_image = tf.reshape(variable, [1, height, width, 1])

            tf.summary.image(name, tensor_as_image, collections=['weight_images'])
        weight_images_summary_op = tf.summary.merge_all('weight_images')

    saver = tf.train.Saver(max_to_keep=2)
    best_validation_saver = tf.train.Saver(max_to_keep=2)

    checkpoint_interval = 0
    if options.mode == 'train':
        checkpoint_interval = 60

    supervisor_saver = best_validation_saver if options.mode == 'decode' else saver

    sv = tf.train.Supervisor(logdir=options.logdir,
                             saver=supervisor_saver,
                             summary_op=None,
                             summary_writer=None,
                             save_model_secs=checkpoint_interval,
                             global_step=model.global_step)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = options.allow_gpu_growth

    with sv.managed_session(config=config) as sess:
        if len(sv.saver.last_checkpoints) > 0:
            print("Restored from saved model '{}'".format(sv.saver.last_checkpoints[-1]))

        training_writer = tf.summary.FileWriter(path.join(options.logdir, 'training'))
        validation_writer = tf.summary.FileWriter(path.join(options.logdir, 'validation'))

        initial_epoch = sess.run(model.epoch)

        run_context = RunContext()

        run_context.options = options
        run_context.model = model
        run_context.session = sess
        run_context.saver = saver
        run_context.best_validation_saver = best_validation_saver
        run_context.training_batcher = training_batcher
        run_context.validation_batcher = validation_batcher
        run_context.decode_batcher = decode_batcher
        run_context.vocabulary = vocabulary
        run_context.word_dict = word_dict
        run_context.word_embedding_dim = word_embedding_dim
        run_context.training_writer = training_writer
        run_context.validation_writer = validation_writer
        run_context.training_summary_op = training_summary_op
        run_context.weight_images_summary_op = weight_images_summary_op
        run_context.collect_run_metadata = options.collect_run_metadata
        run_context.epoch = initial_epoch
        run_context.validation_batch_interval = math.ceil(
            run_context.options.validation_interval / run_context.options.batch_size)

        # Log weight images once before starting (or resuming) training
        if options.mode == 'train' and run_context.options.log_weight_images:
            [weight_images_summary, global_step_value] = sess.run(
                [run_context.weight_images_summary_op, model.global_step])
            run_context.training_writer.add_summary(weight_images_summary, global_step_value)

        try:
            if options.mode == 'train':
                while not sv.should_stop():
                    run_epoch(run_context, training=True)
                    print("Finished epoch {}".format(run_context.epoch))
            elif options.mode == 'validate':
                raise Exception("Validate mode is no longer supported")
                # run_epoch(run_context, training=False)
                print("Finished validating.")
            else:
                run_decode(run_context)
                print("Finished decoding.")

        finally:
            # Manually save after ctrl-c signal
            run_context.training_writer.flush()
            run_context.validation_writer.flush()
            if options.mode == 'train':
                saver.save(sess, sv.save_path, global_step=model.global_step)

                print("Model saved as '{}'".format(sv.saver.last_checkpoints[-1]))


def run_epoch(run_context, training):
    batcher = run_context.training_batcher if training else run_context.validation_batcher
    model = run_context.model
    sess = run_context.session

    if not training:
        total_validation_samples = 0
        total_validation_loss = 0

    performance_count_batches = 0
    performance_count_samples = 0
    performance_count_input_tokens = 0
    logging_timer_start = timer()

    for batch_index, raw_batch in enumerate(batcher.get_batches(shuffle_batches=training)):
        documents, queries, references, entities = raw_batch[:4]

        batch = process_batch(run_context.options, run_context.vocabulary, documents, queries, references, entities)
        actual_batch_size = batch.documents.shape[0]
        document_length = batch.documents.shape[1]

        feed_dict = {
            model.documents_placeholder: batch.documents,
            model.document_lengths_placeholder: batch.document_lengths,
            model.queries_placeholder: batch.queries,
            model.query_lengths_placeholder: batch.query_lengths,
            model.references_placeholder: batch.references,
            model.reference_lengths_placeholder: batch.reference_lengths,
            model.pointer_reference_placeholder: batch.pointer_reference,
            model.pointer_switch_placeholder: batch.pointer_switch
        }

        if training:
            run_options = None
            run_metadata = None
            if run_context.collect_run_metadata:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            _, loss_value, summary, global_step_value = sess.run(
                [model.minimize_operation, model.final_train_loss, run_context.training_summary_op, model.global_step],
                feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

            run_context.training_writer.add_summary(summary, global_step_value)

            if run_context.collect_run_metadata:
                run_context.training_writer.add_graph(sess.graph, global_step_value)
                run_context.training_writer.add_run_metadata(run_metadata, 'train_{}'.format(global_step_value),
                                                             global_step_value)
                # Only collect metadata once
                run_context.collect_run_metadata = False
        else:
            if run_context.options.mode == 'train':
                (loss_value, vocabulary_argmax_value, attention_argmax_values, pointer_enabled_values,
                 global_step_value) = sess.run(
                    [model.main_train_loss, model.train_vocabulary_argmax, model.train_attention_argmax,
                     model.train_pointer_enabled, model.global_step],
                    feed_dict=feed_dict)
            else:
                (loss_value, vocabulary_argmax_value, attention_argmax_values, pointer_enabled_values,
                 global_step_value) = sess.run(
                    [model.main_loss, model.vocabulary_argmax, model.attention_argmax, model.pointer_enabled,
                     model.global_step],
                    feed_dict=feed_dict)
            total_validation_samples += actual_batch_size
            total_validation_loss += loss_value * actual_batch_size

        performance_count_batches += 1
        performance_count_samples += actual_batch_size
        performance_count_input_tokens += actual_batch_size * document_length

        now = timer()
        since_log = now - logging_timer_start
        if since_log >= 10.:
            print("============================")

            if training:
                # Alert if weights grow very large
                variables = tf.trainable_variables()
                for variable in variables:
                    variable_value = sess.run(variable)
                    min_value = np.min(variable_value)
                    max_value = np.max(variable_value)

                    if max(abs(min_value), abs(max_value)) > 100.:
                        print("{}\n  min: {}\n  max: {}".format(variable.name, min_value, max_value))

            run_type = "training" if training else "validation"
            print("Run type: {}".format(run_type))
            print("Epoch: {}, step: {}, batch: {}".format(run_context.epoch + 1,
                                                          global_step_value + 1,
                                                          batch_index + 1))

            epoch_progress = 100 * batch_index / batcher.num_batches
            print("Epoch progress: {0:.2f}%".format(epoch_progress))

            if not training:
                batch_debug_output = list(
                    zip(vocabulary_argmax_value, attention_argmax_values, pointer_enabled_values, documents, queries,
                        references))

                # Only print first item in batch
                batch_debug_output = [batch_debug_output[0]]

                for (output_summary, attention_argmax_value, pointer_enabled_value, document, query,
                     reference) in batch_debug_output:
                    print("Q: {}".format(" ".join(query)))
                    print("I: {}".format(" ".join(document[:20])))
                    print("R: {}".format(" ".join(reference)))
                    output_train_string = vocabularized_to_string(output_summary, attention_argmax_value,
                                                                  pointer_enabled_value, document,
                                                                  run_context.vocabulary)
                    print("O: {}".format(output_train_string))
            print("Loss: {}".format(loss_value))

            ms_per_batch = 1000 * since_log / performance_count_batches
            ms_per_sample = 1000 * since_log / performance_count_samples
            ms_per_input_token = 1000 * since_log / performance_count_input_tokens

            print("Time per batch: {0:.3f} ms".format(ms_per_batch))
            print("Time per sample: {0:.3f} ms".format(ms_per_sample))
            print("Time per input token: {0:.3f} ms".format(ms_per_input_token))

            print("============================")

            performance_count_batches = 0
            performance_count_samples = 0
            performance_count_input_tokens = 0
            logging_timer_start = now

        if training:
            if global_step_value % run_context.validation_batch_interval == 0:
                run_epoch(run_context, training=False)

                performance_count_batches = 0
                performance_count_samples = 0
                performance_count_input_tokens = 0
                logging_timer_start = timer()

    if not training:
        mean_validation_loss = total_validation_loss / total_validation_samples
        mean_validation_loss_summary = tf.Summary(value=[
            tf.Summary.Value(tag="main_loss", simple_value=mean_validation_loss),
        ])
        run_context.validation_writer.add_summary(mean_validation_loss_summary, global_step_value)
        run_context.training_writer.flush()
        run_context.validation_writer.flush()

        # Save checkpoint if validation loss has improved
        best_validation_loss_value = sess.run(model.best_validation_loss)
        if mean_validation_loss < best_validation_loss_value:
            # Update stored best validation
            sess.run(model.best_validation_assign,
                     feed_dict={
                         model.new_best_validation: mean_validation_loss
                     })

            best_validation_path = path.join(run_context.options.logdir, 'best_validation.ckpt')
            run_context.best_validation_saver.save(sess, best_validation_path, global_step=model.global_step)

    if training:
        run_context.epoch = sess.run(model.increment_epoch_op)

        # Log weight matrices only each epoch. It seems that logging them frequently often gives very large
        # events.out.tfevents.* files.
        if run_context.options.log_weight_images:
            weight_images_summary = sess.run(run_context.weight_images_summary_op)
            run_context.training_writer.add_summary(weight_images_summary, global_step_value)


def run_decode(run_context):
    model = run_context.model
    sess = run_context.session
    batcher = run_context.decode_batcher

    initial_last_output = [run_context.vocabulary.word_to_id('<GO>')]

    print("Decoding...")

    for batch_index, raw_batch in enumerate(batcher.get_batches(shuffle_batches=False)):
        documents, queries, document_ids, query_ids = raw_batch

        batch = process_batch(run_context.options, run_context.vocabulary, documents, queries)

        feed_dict = {
            model.documents_placeholder: batch.documents,
            model.document_lengths_placeholder: batch.document_lengths,
            model.queries_placeholder: batch.queries,
            model.query_lengths_placeholder: batch.query_lengths
        }

        encoder_outputs, query_state, next_decoder_state, partial_query_score, partial_encoder_score = sess.run(
            [model.encoder_outputs, model.query_last, model.final_encoder_state, model.query_attention_partial_score,
             model.encoder_state_attention_partial_scores], feed_dict=feed_dict
        )

        beam_width = run_context.options.beam_width
        max_output_length = run_context.options.max_output_length
        num_finished_hypotheses = 0

        finished_hypotheses_probs = []
        finished_hypotheses_voc_argmax = []
        finished_hypotheses_attention_argmax = []
        finished_hypotheses_attention_softmax = []
        finished_hypotheses_pointer_enabled = []

        best_hypotheses = []

        feed_dict = {
            model.documents_placeholder: batch.documents,
            model.document_lengths_placeholder: batch.document_lengths,
            model.queries_placeholder: batch.queries,
            model.query_lengths_placeholder: batch.query_lengths,
            model.beam_width_placeholder: beam_width,
            model.decode_last_output_placeholder: initial_last_output,
            model.initial_decoder_state_placeholder: next_decoder_state,
            model.pre_computed_encoder_states_placeholder: encoder_outputs,
            model.pre_computed_query_state_placeholder: query_state,
            model.query_attention_partial_score_placeholder: partial_query_score,
            model.encoder_state_attention_partial_scores_placeholder: partial_encoder_score
        }

        (best_hypotheses_voc_argmax, best_hypotheses_probs, best_hypotheses_next_decoder_states,
         best_hypotheses_attention_argmax, attention_softmax, pointer_enabled) = sess.run(
            [model.top_k_vocabulary_argmax, model.top_k_probabilities, model.one_step_decoder_state,
             model.attention_argmax, model.attention_softmax, model.pointer_enabled],
            feed_dict=feed_dict)

        for i in range(beam_width):
            best_hypotheses.append(Hypothesis(best_hypotheses_probs[0, i],
                                              [best_hypotheses_voc_argmax[0, i]],
                                              [best_hypotheses_attention_argmax[0]],
                                              best_hypotheses_next_decoder_states[0],
                                              [attention_softmax[0]],
                                              [pointer_enabled[0]]))

        time_step = 1
        while num_finished_hypotheses < beam_width:

            hypotheses = []

            for hypothesis in best_hypotheses:
                attended_token = batch.documents[0, hypothesis.attention_argmax[-1]]
                vocabulary_token = hypothesis.voc_argmax[-1]
                last_pointer_enabled = hypothesis.pointer_enabled[-1]
                last_output_index = attended_token if last_pointer_enabled == 1 else vocabulary_token

                feed_dict = {
                    model.documents_placeholder: batch.documents,
                    model.document_lengths_placeholder: batch.document_lengths,
                    model.queries_placeholder: batch.queries,
                    model.query_lengths_placeholder: batch.query_lengths,
                    model.beam_width_placeholder: beam_width,
                    model.decode_last_output_placeholder: [last_output_index],
                    model.initial_decoder_state_placeholder: [hypothesis.next_decoder_state],
                    model.pre_computed_encoder_states_placeholder: encoder_outputs,
                    model.pre_computed_query_state_placeholder: query_state,
                    model.query_attention_partial_score_placeholder: partial_query_score,
                    model.encoder_state_attention_partial_scores_placeholder: partial_encoder_score
                }

                (top_k_voc_argmax, top_k_probs, next_decoder_state, attention_argmax, attention_softmax,
                 pointer_enabled) = sess.run(
                    [model.top_k_vocabulary_argmax, model.top_k_probabilities, model.one_step_decoder_state,
                     model.attention_argmax, model.attention_softmax, model.pointer_enabled],
                    feed_dict=feed_dict)

                for j in range(beam_width):
                    new_prob = hypothesis.probability * top_k_probs[0, j]
                    new_voc_argmax = hypothesis.voc_argmax + [top_k_voc_argmax[0, j]]
                    new_attention_argmax = hypothesis.attention_argmax + [attention_argmax[0]]
                    new_attention_softmax = hypothesis.attention_softmax + [attention_softmax[0]]
                    new_pointer_enabled = hypothesis.pointer_enabled + [pointer_enabled[0]]
                    hypotheses.append(Hypothesis(
                        new_prob, new_voc_argmax, new_attention_argmax, next_decoder_state[0], new_attention_softmax,
                        new_pointer_enabled))

            hypotheses.sort(key=lambda hyp: hyp.probability, reverse=True)

            added_to_next_beam = 0
            best_hypotheses = []

            num_considered_hypotheses = 0
            while num_finished_hypotheses < beam_width and added_to_next_beam < beam_width:

                hypothesis = hypotheses[num_considered_hypotheses]

                if hypothesis.voc_argmax[-1] == run_context.vocabulary.word_to_id(
                        '<EOS>') or time_step + 1 >= max_output_length:
                    num_finished_hypotheses += 1
                    finished_hypotheses_probs.append(hypothesis.probability)
                    finished_hypotheses_voc_argmax.append(hypothesis.voc_argmax)
                    finished_hypotheses_attention_argmax.append(hypothesis.attention_argmax)
                    finished_hypotheses_attention_softmax.append(hypothesis.attention_softmax)
                    finished_hypotheses_pointer_enabled.append(hypothesis.pointer_enabled)
                else:
                    best_hypotheses.append(hypothesis)
                    added_to_next_beam += 1

                num_considered_hypotheses += 1
            time_step += 1

        final_hypothesis_prob = -np.inf

        for i in range(beam_width):
            if finished_hypotheses_probs[i] > final_hypothesis_prob:
                final_hypothesis_prob = finished_hypotheses_probs[i]
                final_hypothesis_voc_argmax = finished_hypotheses_voc_argmax[i]
                final_hypothesis_attention_argmax = finished_hypotheses_attention_argmax[i]
                final_hypothesis_attention_softmax = finished_hypotheses_attention_softmax[i]
                final_hypothesis_pointer_enabled = finished_hypotheses_pointer_enabled[i]

        write_output_summaries(run_context,
                               [np.array(final_hypothesis_voc_argmax)],
                               [np.array(final_hypothesis_attention_argmax)],
                               [np.array(final_hypothesis_pointer_enabled)],
                               documents[0], document_ids, query_ids)

        document_query_id = '{}.{}.txt'.format(document_ids[0], query_ids[0])

        softmax_path = path.join(run_context.options.decode_out_dir, 'attention_softmax')
        if not path.isdir(softmax_path):
            os.makedirs(softmax_path)
        np.savetxt(path.join(softmax_path, document_query_id), final_hypothesis_attention_softmax)

        output_prob_path = path.join(run_context.options.decode_out_dir, 'output_probabilities')
        if not path.isdir(output_prob_path):
            os.makedirs(output_prob_path)
        np.savetxt(path.join(output_prob_path, document_query_id), [final_hypothesis_prob])


def process_batch(options, vocabulary, document_batch, query_batch, reference_batch=None, entities_batch=None):
    # Limit document size
    max_document_tokens = 800
    document_batch = [document[:max_document_tokens] for document in document_batch]

    # Add special <EOS> symbol
    for batch_texts in [query_batch, reference_batch]:
        if batch_texts is not None:
            for text in batch_texts:
                text.append('<EOS>')

    documents = vocabularize_batch(document_batch, vocabulary)
    queries = vocabularize_batch(query_batch, vocabulary)
    if reference_batch is not None:
        references, pointer_reference, pointer_switch = process_references(document_batch, reference_batch,
                                                                           entities_batch, vocabulary,
                                                                           options.target_vocabulary_size)
    actual_batch_size = len(document_batch)
    processed_batch_elements = []
    if reference_batch is None:
        loop_elements = (documents, queries)
    else:
        loop_elements = (documents, queries, references, pointer_reference, pointer_switch)
    for texts in loop_elements:
        data_shape = list(texts[0].shape[1:])

        longest_text_length = max(map(len, texts))
        padded = np.zeros([actual_batch_size, longest_text_length] + data_shape, dtype=np.int32)
        lengths = np.zeros([actual_batch_size], dtype=np.int32)

        for index, text in enumerate(texts):
            length = np.size(text, 0)
            padded[index, :length] = text
            lengths[index] = length

        processed_batch_elements.append(padded)
        processed_batch_elements.append(lengths)
    batch = ProcessedBatch(*processed_batch_elements)
    return batch


def process_references(document_batch, reference_batch, entities_batch, vocabulary, target_vocabulary_size):
    batch_size = len(document_batch)
    batched_reference = [None] * batch_size
    batched_pointer_reference = [None] * batch_size
    batched_pointer_switch = [None] * batch_size

    pad_id = vocabulary.word_to_id('<PAD>')
    unk_id = vocabulary.word_to_id('<UNK>')

    for batch_index, (document, reference, entities) in enumerate(zip(document_batch, reference_batch, entities_batch)):
        reference_length = len(reference)
        vocabularized_reference = vocabularize(reference, vocabulary, target_vocabulary_size - 1)
        pointer_reference = np.zeros([reference_length], dtype=np.int32)
        pointer_switch = np.zeros([reference_length], dtype=np.int32)

        # Build index to look up words in document
        document_index = defaultdict(list)
        for index, token in enumerate(document):
            document_index[token].append(index)

        # Sort to find longest entities first (e.g. 'President of America' before 'America')
        entities.sort(key=len, reverse=True)
        pointed_reference_entity_indices = set()
        for entity_tokens in entities:

            # Find the first occurrence of the whole entity in the reference
            for ref_start_index in range(len(reference)):
                ref_end_index = ref_start_index + len(entity_tokens)

                # Make sure that entities in the reference don't overlap
                if any([index in pointed_reference_entity_indices for index in range(ref_start_index, ref_end_index)]):
                    continue

                if entity_tokens == reference[ref_start_index:ref_end_index]:
                    # Find the first occurrence of the entity in the document
                    for doc_start_index in range(len(document)):
                        doc_end_index = doc_start_index + len(entity_tokens)
                        if entity_tokens == document[doc_start_index:doc_end_index]:
                            for entity_index in range(len(entity_tokens)):
                                reference_entity_index = ref_start_index + entity_index
                                pointed_reference_entity_indices.add(reference_entity_index)
                                pointer_switch[reference_entity_index] = 1
                                pointer_reference[reference_entity_index] = doc_start_index + entity_index
                                vocabularized_reference[reference_entity_index] = unk_id
                            break

        # Set pointer for words not in summary vocabulary
        for index, token in enumerate(reference):
            if vocabularized_reference[index] == unk_id and pointer_switch[index] == 0:
                occurrences = document_index.get(token)
                if occurrences is not None:
                    pointer_switch[index] = 1
                    pointer_reference[index] = occurrences[0]
                else:
                    # To avoid teaching the model to output <unk>, loss is masked for these time steps
                    vocabularized_reference[index] = pad_id

        batched_reference[batch_index] = vocabularized_reference
        batched_pointer_reference[batch_index] = pointer_reference
        batched_pointer_switch[batch_index] = pointer_switch
    return batched_reference, batched_pointer_reference, batched_pointer_switch


def write_output_summaries(run_context, vocabularized_summaries, batched_attention_indices, batched_pointer_enabled,
                           document, document_ids, query_ids):
    summary_dir = path.join(run_context.options.decode_out_dir, 'summaries')

    if not path.isdir(summary_dir):
        os.makedirs(summary_dir)
    output_summaries = list(zip(vocabularized_summaries, batched_attention_indices, batched_pointer_enabled,
                                document_ids, query_ids))
    for vocabularized_summary, attention_indices, pointer_enabled, document_id, query_id in output_summaries:
        out_name = '{}.{}.txt'.format(document_id, query_id)
        out_path = path.join(summary_dir, out_name)
        summary = vocabularized_to_string(vocabularized_summary, attention_indices, pointer_enabled, document,
                                          run_context.vocabulary)
        with io.open(out_path, 'w', encoding='utf-8') as file:
            file.write(summary)


def vocabularize(text, vocabulary, max_id=np.inf):
    unk_id = vocabulary.word_to_id('<UNK>')
    vocabularized = [vocabulary.word_to_id(word) for word in text]
    vocabularized = [id_ if id_ <= max_id else unk_id for id_ in vocabularized]
    return np.array(vocabularized, dtype=np.int32)


def vocabularize_batch(texts, vocabulary, max_id=np.inf):
    return [vocabularize(text, vocabulary, max_id) for text in texts]


def vocabularized_to_string(vocabularized, attention_indices, pointer_enabled, document, vocabulary):
    eos_id = vocabulary.word_to_id('<EOS>')
    python_list = vocabularized.tolist()
    try:
        end_index = python_list.index(eos_id)
    except ValueError:
        end_index = len(python_list)

    out_tokens = []

    for index, token in enumerate(python_list[:end_index]):
        if pointer_enabled[index] == 1:
            out_token = document[attention_indices[index]]
            print("Pointer replace: {}".format(out_token))
        else:
            out_token = vocabulary.id_to_word(token)
        out_tokens.append(out_token)

    return " ".join(out_tokens)


if __name__ == '__main__':
    tf.app.run()
