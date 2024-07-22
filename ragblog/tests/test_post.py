from ragblog.post import Post


def test_basic():
    post = Post(title="not_title", text="not_text")
    assert str(post) == """{"title": "not_title", "text": "not_text"}"""


def test_from_url():
    post = Post.from_url(
        url="https://delightfulobservaciones.blogspot.com/2019/06/poco-antes-de-helena.html"
    )
    assert post.title == "Poco antes de Helena"
    assert len(post.text) > 0
