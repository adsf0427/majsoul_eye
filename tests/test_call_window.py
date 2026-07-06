"""Tests for is_call_window frame-drop predicate.

A call event (chi/pon/kan/nukidora) has arrived but the forced follow-up dahai
hasn't: the call animation is mid-flight, so GT leads the pixels (river already
shrunk, meld already added). Frame-level drop, same policy as is_deal_window.
"""

from majsoul_eye.state.replay import BoardState, is_call_window


def test_is_call_window_true_on_call_events():
    """All 6 call event types return True."""
    for event_type in ("chi", "pon", "daiminkan", "ankan", "kakan", "nukidora"):
        s = BoardState(last_event=event_type, in_round=True)
        assert is_call_window(s) is True, f"Failed for event_type={event_type}"


def test_is_call_window_false_on_non_call_events():
    """Non-call events return False."""
    for event_type in ("dahai", "tsumo", "start_kyoku", "reach", "reach_accepted"):
        s = BoardState(last_event=event_type, in_round=True)
        assert is_call_window(s) is False, f"Failed for event_type={event_type}"


def test_is_call_window_false_on_none_last_event():
    """None last_event returns False."""
    s = BoardState(last_event=None, in_round=True)
    assert is_call_window(s) is False


def test_is_call_window_false_no_last_event_attr():
    """Object without last_event attribute returns False (duck typing)."""
    # Plain dict-like object with no last_event
    state_dict = {"in_round": True}
    assert is_call_window(state_dict) is False


def test_is_call_window_with_duck_typed_object():
    """Duck-typed object with last_event works."""
    class DuckState:
        def __init__(self, last_event):
            self.last_event = last_event

    for event_type in ("chi", "pon", "daiminkan", "ankan", "kakan", "nukidora"):
        s = DuckState(last_event=event_type)
        assert is_call_window(s) is True, f"Failed for event_type={event_type}"


def test_is_call_window_false_duck_typed_non_call():
    """Duck-typed object with non-call event returns False."""
    class DuckState:
        def __init__(self, last_event):
            self.last_event = last_event

    s = DuckState(last_event="dahai")
    assert is_call_window(s) is False


def test_is_call_window_true_on_bundled_record():
    """Regression: a real record can bundle a call with another event (e.g.
    [pon, dora]) — last_event is overwritten to "dora" but last_event_types
    still carries "pon", so the window must still be flagged True."""
    s = BoardState(last_event="dora", last_event_types=frozenset({"pon", "dora"}), in_round=True)
    assert is_call_window(s) is True


def test_is_call_window_false_when_types_and_last_event_both_empty():
    """No event applied at all this record: last_event_types empty, last_event
    None -> False (not just "falls through by accident")."""
    s = BoardState(last_event=None, last_event_types=frozenset(), in_round=True)
    assert is_call_window(s) is False


if __name__ == "__main__":
    test_is_call_window_true_on_call_events()
    test_is_call_window_false_on_non_call_events()
    test_is_call_window_false_on_none_last_event()
    test_is_call_window_false_no_last_event_attr()
    test_is_call_window_with_duck_typed_object()
    test_is_call_window_false_duck_typed_non_call()
    test_is_call_window_true_on_bundled_record()
    test_is_call_window_false_when_types_and_last_event_both_empty()
    print("All tests passed!")
