import numpy as np

from slam.bow_loop_closure import BoWLoopClosure


def test_bow_similarity_is_symmetric_for_sparse_histograms():
    a = {1: 0.8, 3: 0.6}
    b = {1: 0.5, 2: 0.5}
    assert BoWLoopClosure.similarity(a, a) == 1.0
    assert BoWLoopClosure.similarity(a, b) == BoWLoopClosure.similarity(b, a)


def test_bow_similarity_ignores_non_overlapping_words():
    assert BoWLoopClosure.similarity({1: 1.0}, {2: 1.0}) == 0.0


def test_vocabulary_trains_and_query_ranks_matching_keyframe_first():
    rng = np.random.default_rng(17)
    descriptors = rng.integers(0, 256, size=(96, 32), dtype=np.uint8)
    other = rng.integers(0, 256, size=(96, 32), dtype=np.uint8)
    bow = BoWLoopClosure(words=16, min_interval=1, min_skip_frames=0, training_frames=2)
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


def test_default_vocabulary_waits_for_multiple_views():
    rng = np.random.default_rng(23)
    bow = BoWLoopClosure(words=16, min_interval=1, min_skip_frames=0)
    for frame_idx in range(9):
        descriptors = rng.integers(0, 256, size=(64, 32), dtype=np.uint8)
        bow.register(frame_idx, object(), descriptors)
    assert bow.vocabulary.centers is None
    assert len(bow._pending) == 9

    descriptors = rng.integers(0, 256, size=(64, 32), dtype=np.uint8)
    bow.register(9, object(), descriptors)
    assert bow.vocabulary.centers is not None
    assert bow.vocabulary.centers.shape == (16, 256)
    assert len(bow._pending) == 0
