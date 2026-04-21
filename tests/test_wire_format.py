"""Tests for wire format encode/decode."""

from maayatrain.comms.wire_format import MsgKind, decode_bytes, encode, new_peer_id


def test_roundtrip_no_payload():
    """Encode and decode a message with no payload."""
    pid = new_peer_id()
    raw = encode(MsgKind.HEARTBEAT, pid)
    frame = decode_bytes(raw)

    assert frame.kind == MsgKind.HEARTBEAT
    assert frame.sender_id == pid
    assert frame.payload == b""
    assert frame.timestamp > 0


def test_roundtrip_with_payload():
    """Encode and decode a message with binary payload."""
    pid = "test-peer-001"
    payload = b"hello world gradient data"
    raw = encode(MsgKind.SYNC_GRADIENTS, pid, payload, compression="fp16_gzip")
    frame = decode_bytes(raw)

    assert frame.kind == MsgKind.SYNC_GRADIENTS
    assert frame.sender_id == pid
    assert frame.payload == payload
    assert frame.compression == "fp16_gzip"


def test_roundtrip_with_extra():
    """Extra fields are preserved in the header."""
    pid = "test-peer-002"
    raw = encode(MsgKind.HANDSHAKE, pid, extra={"model": "gpt2-small", "step": 5000})
    frame = decode_bytes(raw)

    assert frame.kind == MsgKind.HANDSHAKE
    assert frame.header["model"] == "gpt2-small"
    assert frame.header["step"] == 5000


def test_all_msg_kinds():
    """All MsgKind values can be encoded and decoded."""
    pid = "test-peer-003"
    for kind in MsgKind:
        raw = encode(kind, pid)
        frame = decode_bytes(raw)
        assert frame.kind == kind


def test_peer_id_uniqueness():
    """Generated peer IDs are unique."""
    ids = {new_peer_id() for _ in range(100)}
    assert len(ids) == 100


def test_large_payload():
    """Can handle payloads up to 10 MB."""
    pid = "test-peer-004"
    payload = b"x" * (10 * 1024 * 1024)  # 10 MB
    raw = encode(MsgKind.MODEL_WEIGHTS, pid, payload)
    frame = decode_bytes(raw)
    assert len(frame.payload) == 10 * 1024 * 1024
