from slam.bow_loop_closure import BoWLoopClosure

def test_bow_similarity_is_symmetric_for_sparse_histograms():
    a = {1: 0.8, 3: 0.6}
    b = {1: 0.8, 3: 0.6}
    assert BoWLoopClosure.similarity(a, b) == 1.0

def test_bow_similarity_ignores_non_overlapping_words():
    assert BoWLoopClosure.similarity({1: 1.0}, {2: 1.0}) == 0.0
