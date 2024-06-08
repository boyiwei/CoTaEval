import redis
import numpy as np
import dataportraits.utils as utils
import dataportraits.code_proc as code_proc
import struct
import math
import sys
import json
import contextlib
from collections import namedtuple

# struct info to decode redis headers
BFInfo = namedtuple("BFInfo", "chain_size chain_n_filters options growth bytes bits current_size error bits_per_element hashes max_entries n2")
BF_INFO_STRUCT = "=QLLLQQQddLQB"

def check_chain(membership_tests, index, step, accumulator):
    if index >= len(membership_tests):
        return accumulator

    is_member = membership_tests[index][0]
    if is_member:
        accumulator.append(index)
        return check_chain(membership_tests, index + step, step, accumulator)

    return accumulator


def chain_overlaps(membership_tests, width):
    already_found = set()
    matches = []
    idxs = []
    for ix, (_, _) in enumerate(membership_tests):
        if ix in already_found:
            continue

        run_idxs = check_chain(membership_tests, ix, step=width, accumulator=[])

        if len(run_idxs) == 0:
            continue

        run_segments =  [membership_tests[i][1] for i in run_idxs]
        matches.append(run_segments)
        idxs.append(run_idxs)
        already_found.update(run_idxs)

    return matches, idxs


def build_text_pipeline_fn(**kwargs):
    apply_code_processor = kwargs.get('apply_code_processor', False)

    # mock a tokenizer object
    def tokenizer_fn(batches_of_text, **kwargs):
        tokenizer_fn.pad_token_id = None
        return {'input_ids' : batches_of_text}

    if apply_code_processor:
        def pre_process(text):
            return code_proc.proc_code(text)
    else:
        # print("[WARNING] not using code proc", file=sys.stderr)
        def pre_process(text):
            return text

    # get parameters
    stride = kwargs.get('stride')
    width = kwargs.get('width')

    def pipeline(batches_of_text):
        batches_of_text = [pre_process(text) for text in batches_of_text]
        tokens = tokenizer_fn(batches_of_text)['input_ids']

        results = []
        for instance in tokens:
            ngrams = utils.chunk_sequence_strided(instance, width, stride, stop_token=None)
            results.append(ngrams)
        return results
    return pipeline

class RedisBFSketch:

    def __init__(self, host, port, key, width):
        self.host = host
        self.port = port
        self.redis_client = redis.Redis(host=host, port=port)
        self.bf_client = self.redis_client.bf()
        self.key = key
        self.query_batch_size = 10000
        self.width = width
        self.self_test()

    def chunk(self, batch_of_strings, stride):
        text_pipeline = build_text_pipeline_fn(width=self.width, stride=stride, apply_code_processor=False)
        return utils.flatten_batched(text_pipeline(batches_of_text=batch_of_strings))


    def contains_from_text(self, documents, stride=1, sort_chains_by_length=True):
        # Here we added an options of customizing stride, for character level detection, stride is set 1 as default; for token level detection, stride=5
        lens, segments = self.chunk(documents, stride)
        results = self.contains_all(segments)

        membership_results = utils.unflatten(lens, zip(results, segments), empty_element = [[], []])
        assert len(membership_results) == len(documents), "Didn't get membership results for some document"

        outputs = []
        for doc_num, (original_document, segment_memberships) in enumerate(zip(documents, membership_results)):

            # ensure the preprocessed document is available
            original_document = code_proc.proc_code(original_document)

            doc_report = {
                'idx' : doc_num,
                'doc' : original_document,
                'segments' : [],
                'is_member' : [],
                'chains' : [],
                'chain_idxs' : []
            }

            if len(segment_memberships[0]) > 0:
                membership_tests, segments = zip(*segment_memberships) # unzip
                doc_report['chains'], doc_report['chain_idxs'] = chain_overlaps(segment_memberships, self.width)
                doc_report['segments'] = segments
                doc_report['is_member'] = [bool(i) for i in membership_tests]
                if sort_chains_by_length:
                    assert all((len(a) == len(b)) for a, b in zip(doc_report['chains'], doc_report['chain_idxs']))
                    doc_report['chains'] = sorted(doc_report['chains'], key = lambda x : len(x), reverse=True) # these are sorted the same way because each element has the same length. asserted above.
                    doc_report['chain_idxs'] = sorted(doc_report['chain_idxs'], key = lambda x : len(x), reverse=True)

            outputs.append(doc_report)

        return outputs

    def contains(self, item):
        return self.contains_all([item])[0]

    def contains_all(self, items):
        self.exists()
        results = []
        for batch in utils.batcher_fn(items, self.query_batch_size):
            results.extend(self.bf_client.mexists(self.key, *batch))
        return results

    def self_test(self):
        self.redis_client.ping()

    def exists(self):
        assert self.redis_client.exists(self.key) == 1, f"Key `{self.key}` doesn't exist in the specified server"

    def stats(self):
        self.exists()
        info_bytes = self._scandump(self.key, 0)[1]
        return BFInfo._make(struct.unpack(BF_INFO_STRUCT, info_bytes))

    def _scandump(self, key, iter):
        #monkey patch to bypass hiredis issue
        #https://github.com/redis/redis-py/blob/936d49f4c1dd6cf0c2e3ad80de29f25eef81d8a9/redis/commands/bf/commands.py
        params = [key, iter]
        options = {}
        options[redis.client.NEVER_DECODE] = []
        return self.redis_client.execute_command(redis.commands.bf.BF_SCANDUMP, *params, **options)


    def iter_dump(self, verbose=True, return_iter=False):
        #dump the raw bytes. https://redis.io/commands/bf.scandump/

        with utils.get_progress(unit_scale=True, unit='B') if verbose else contextlib.nullcontext() as progress:
            iter = 0
            while True:
                iter, data = self._scandump(self.key, iter)
                if iter == 0:
                    return
                else:
                    if verbose:
                        progress.update(len(data))
                    if return_iter:
                        yield iter, data
                    else:
                        yield data

    def to_file(self, path, verbose=False):
        idxs = []
        with open(path, 'wb') as f:
            for it, (iter, block) in enumerate(self.iter_dump(return_iter=True, verbose=verbose)):
                idx = {'iter' : iter, 'block_num' : it, 'block_size' : len(block)}
                idxs.append(idx)
                f.write(block)

        if path == '/dev/null':
            return

        with open(path + '.idx', 'w') as f:
            json.dump(idxs, f, indent=2)

    @classmethod
    def from_file(cls, host, port, key, width, path, overwrite=False):
        with open(path + '.idx') as f:
            idxs = json.load(f)

        bf = cls(host, port, key, width)

        if bf.redis_client.exists(key):
            if overwrite:
                bf.redis_client.delete(key)
            else:
                raise Exception(f"Redis instance already contains key: {key}")

        with open(path, 'rb') as f:
            for idx in idxs:
                # print(idx['block_num'])
                data = f.read(idx['block_size'])
                bf.bf_client.loadchunk(key, idx['iter'], data)
        return bf

    def count_bits(self):
        dump = self.iter_dump()
        _ = BFInfo._make(struct.unpack(BF_INFO_STRUCT, next(dump)))
        set_bits = 0
        for block in dump:
            set_bits += utils.sum_bits_from_packed(np.frombuffer(block, dtype=np.uint8))
        return set_bits

    def approximate_size(self):
        # Bloom Filters are initialized with an estimated number of elements
        # You can also work backwards and use the load factor (number of set bits)
        # to approximate the number of inserted items
        bf_info = self.stats()
        num_bits = bf_info.bits
        num_hashes = bf_info.hashes

        set_bits = self.count_bits() # this can take a long time
        p_1 = set_bits / num_bits # probability that a bit is 1

        return int(-(num_bits / num_hashes) * math.log(1 - p_1))

    def __repr__(self):
        stats = self.stats()
        dir = self.redis_client.config_get('dir')['dir']
        return f"{dir}@{self.host}:{self.port} [{self.key}={stats.current_size}]"

    def expected_strided_matches(self, sequence):
        """expected_strided_matches.
        Returns the number of expected matches if `sequence` was embedded exactly once in a corpus.
        Accounts for strided n-grams and alignment issues, but doesn't account for document boundaries or false positives.
        :param sequence:

        Unit is chunk
        """
        if len(sequence) < self.width:
            return 0
        return (len(sequence) - (self.width - 1)) / self.width

if __name__ == '__main__':
    # sketch = RedisBFSketch('r4n05', 8899, 'pile.str.code.tight.50-50.bf', 50)
    sketch = RedisBFSketch('r4n05', 8899, 'stack.march-no-pii.tight-2.code.50-50.bf', 50)

    from datasets import load_dataset
    human_eval = load_dataset("openai_humaneval")
    solutions = human_eval['test']['canonical_solution']

    import time
    start = time.time()
    reports = sketch.contains_from_text(solutions, sort_chains_by_length=True)
    end = time.time()
    print(f"Membership testing took {end-start:.4f} seconds")

    # overlap analysis
    expected_chains = 0
    observed_chains = 0
    bad_examples = []
    for original, report in zip(solutions, reports):
        expected = sketch.expected_strided_matches(report['doc'])
        if len(report['chain_idxs']) == 0:
            observed = 0
        else:
            observed = max(len(chain) for chain in report['chain_idxs'])

        if expected != 0:
            score = observed / expected
            bad_examples.append((score, original))

        expected_chains += expected
        observed_chains += observed

    print(f"Overlap pct {observed_chains/expected_chains:.4f}")

    for _, doc in sorted(bad_examples, reverse=True)[:5]:
        print(doc)
        print('-' * 80)
