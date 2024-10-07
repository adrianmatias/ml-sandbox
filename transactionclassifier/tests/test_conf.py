from tests.fixture_set import conf


def test_basic(conf):
    print(conf.json())
    assert conf.api.host == "0.0.0.0"
    assert conf.api.port == 8000
