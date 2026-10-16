from app import checksum


def test_checksum_is_hex():
    assert len(checksum(b"x")) == 32
