from xgboost_distribution.utils import hide_attributes


class MyClass:
    def __init__(self, my_attr):
        self.my_attr = my_attr


def test_hide_attributes():
    my_cls = MyClass(my_attr="fun")

    with hide_attributes(my_cls, ["my_attr"]):
        assert my_cls.__dict__ == {}

    assert my_cls.__dict__ == {"my_attr": "fun"}
