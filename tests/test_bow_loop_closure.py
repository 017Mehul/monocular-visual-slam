import numpy as np

from slam.bow_loop_closure import BoWLoopClosure


def test_bow_similarity_is_symmetric_for_sparse_histograms():
    a = {1: 0.8, 3: 0.6}
    b = {1: 0.8, 3: 0.6}
    assert BoWLoopClosure.similarity(a, b) == 1.0
    assert BoWLoopClosure.similarity(a, b) == BoWLoopClosure.similarity(b, a)


def test_bow_similarity_ignores_non_overlapping_words():
    assert BoWLoopClosure.similarity({1: 1.0}, {2: 1.0}) == 0.0


def test_vocabulary_trains_and_query_ranks_matching_keyframe_first():
    rng = np.random.default_rng(17)
    descriptors = rng.integers(0, 256, size=(96, 32), dtype=np.uint8)
    other = rng.integers(0, 256, size=(96, 32), dtype=np.uint8)
    bow = BoWLoopClosure(words=16, min_interval=1, min_skip_frames=0)
    matching_keyframe = object()
    unrelated_keyframe = object()

    bow.register(0, matching_keyframe, descriptors)
    bow.register(20, unrelated_keyframe, other)

    assert bow.vocabulary.centers is not None
    assert bow.vocabulary.centers.shape == (16, 256)
    results = bow.query(50, descriptors, top_k=2)
    assert len(results) == 2
    assert results[0][1] is matching_keyframe
    assert results[0][0] >= results[1][0]
