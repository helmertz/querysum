import io
import os
import random
import re
from os import path

reference_pattern = re.compile('(.)\.(\d+)\.(\d+)\.txt')
query_pattern = re.compile('()(\d+)\.(\d+)\.txt')  # Starts with empty capture group for similarity and code reusability


class Sample:
    def __init__(self, filename, document_length):
        self.filename = filename
        self.document_length = document_length


def reference_sort_key(filename):
    reference_char, document_id, query_id = reference_pattern.search(filename).groups()
    return int(document_id), int(query_id), reference_char


def query_sort_key(filename):
    reference_char, document_id, query_id = query_pattern.search(filename).groups()
    return int(document_id), int(query_id), reference_char


def read_samples(use_precomputed_lengths, root_dir, sample_directory, document_dir, synthetic,
                 max_count, reference_looping=True):
    precomputed_filename = 'input_lengths.txt' if reference_looping else 'input_lengths_no_reference.txt'
    precomputed_lengths_path = path.join(root_dir, precomputed_filename)

    if use_precomputed_lengths and path.isfile(precomputed_lengths_path):
        with io.open(precomputed_lengths_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        samples = [Sample(*line.split()) for line in lines]
    else:
        samples = []
        num_files_processed = 0

        pattern = reference_pattern if reference_looping else query_pattern

        for sample_filename in os.listdir(sample_directory):
            sample_path = path.join(sample_directory, sample_filename)
            if path.isfile(sample_path):
                _, document_id, query_id = pattern.search(sample_filename).groups()

                document_text = read_document(document_dir, document_id, query_id, synthetic)

                samples.append(Sample(sample_filename, document_text.count(' ') + 1))

                num_files_processed += 1
                if num_files_processed % 10000 == 0:
                    print("{} files processed...".format(num_files_processed))

    # Sort to make batcher deterministic, except when explicitly shuffling
    sort_key = reference_sort_key if reference_looping else query_sort_key
    samples.sort(key=lambda sample: sort_key(sample.filename))

    # Cut off some samples if requested
    if max_count is not None:
        samples = samples[:max_count]

    # Sort by input length
    samples.sort(key=lambda sample: sample.document_length)
    return samples


def read_document(document_dir, document_id, query_id, synthetic):
    if not synthetic:
        document_filename = '{}.txt'.format(document_id)
    else:
        document_filename = '{}.{}.txt'.format(document_id, query_id)
    document_path = path.join(document_dir, document_filename)
    with io.open(document_path, 'r', encoding='utf-8') as document_file:
        document_text = document_file.read()
    return document_text


class Batcher:
    def __init__(self, directory, batch_size, synthetic, max_count=None, reference_looping=True):
        self.document_dir = path.join(directory, 'documents' if not synthetic else 'baselines')
        self.query_dir = path.join(directory, 'queries')
        self.reference_dir = path.join(directory, 'references' if not synthetic else 'synthetic_summaries')
        self.entities_dir = path.join(directory, 'entities')
        self.synthetic = synthetic
        self.reference_looping = reference_looping

        use_precomputed_lengths = not synthetic

        print("Reading document lengths...")
        sample_dir = self.reference_dir if reference_looping else self.query_dir
        sample_infos = read_samples(use_precomputed_lengths, directory, sample_dir, self.document_dir, synthetic,
                                    max_count, reference_looping)
        print("Done reading document lengths!")

        self.batches = []
        for i in range(0, len(sample_infos), batch_size):
            self.batches.append(sample_infos[i:i + batch_size])

        self.num_batches = len(self.batches)

    def get_batches(self, shuffle_batches=False):
        shuffled_batches = list(self.batches)

        if shuffle_batches:
            random.shuffle(shuffled_batches)

        for sample_infos in shuffled_batches:
            batched_documents = []
            batched_queries = []
            if self.reference_looping:
                batched_references = []
                batched_entities = []
            document_ids = []
            query_ids = []
            if self.reference_looping:
                reference_chars = []

            for sample in sample_infos:
                filename = sample.filename
                pattern = reference_pattern if self.reference_looping else query_pattern

                reference_char, document_id, query_id = pattern.search(filename).groups()

                document_text = read_document(self.document_dir, document_id, query_id, self.synthetic)

                query_filename = '{}.{}.txt'.format(document_id, query_id)
                query_path = path.join(self.query_dir, query_filename)
                with io.open(query_path, 'r', encoding='utf-8') as query_file:
                    query_text = query_file.read()

                document_tokens = document_text.split()
                query_tokens = query_text.split()

                batched_documents.append(document_tokens)
                batched_queries.append(query_tokens)

                document_ids.append(document_id)
                query_ids.append(query_id)

                if self.reference_looping:
                    reference_path = path.join(self.reference_dir, filename)
                    entities_filename = '{}.txt'.format(document_id)
                    entities_path = path.join(self.entities_dir, entities_filename)

                    try:
                        with io.open(entities_path, 'r', encoding='utf-8') as entities_file:
                            entities_text = entities_file.read()
                            entities = [entity_row.split(' ') for entity_row in entities_text.splitlines()]
                    except IOError:
                        entities = []

                    with io.open(reference_path, 'r', encoding='utf-8') as reference_file:
                        reference_text = reference_file.read()

                    reference_tokens = reference_text.split()
                    batched_references.append(reference_tokens)
                    batched_entities.append(entities)
                    reference_chars.append(reference_char)

            if self.reference_looping:
                yield batched_documents, batched_queries, batched_references, batched_entities, \
                      document_ids, query_ids, reference_chars
            else:
                yield batched_documents, batched_queries, document_ids, query_ids
