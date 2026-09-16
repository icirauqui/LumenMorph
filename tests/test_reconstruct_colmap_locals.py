from backend.reconstruct_colmap_locals import _window_starts


def test_window_starts_include_a_tail_aligned_window():
    assert _window_starts(180, 15, 10) == [
        0, 10, 20, 30, 40, 50, 60, 70, 80,
        90, 100, 110, 120, 130, 140, 150, 160, 165,
    ]


def test_window_starts_exact_stride_endpoint_is_not_duplicated():
    assert _window_starts(35, 15, 10) == [0, 10, 20]
