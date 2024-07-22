from ragblog.conf import CONF


def test_conf():

    assert CONF.path.chroma == "chroma"
