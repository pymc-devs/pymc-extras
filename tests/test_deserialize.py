import pytest

from pymc_extras.deserialize import (
    DESERIALIZERS,
    DeserializableError,
    deserialize,
    register_deserialization,
)


@pytest.mark.parametrize(
    "unknown_data",
    [
        {"unknown": 1},
        {"dist": "Normal", "kwargs": {"something": "else"}},
        1,
    ],
    ids=["unknown_structure", "prior_like", "non_dict"],
)
def test_unknown_type_raises(unknown_data) -> None:
    match = "Couldn't deserialize"
    with pytest.raises(DeserializableError, match=match):
        deserialize(unknown_data)


class ArbitraryObject:
    def __init__(self, code: str):
        self.code = code
        self.value = 1


@pytest.fixture
def register_arbitrary_object():
    register_deserialization(
        is_type=lambda data: data.keys() == {"code"},
        deserialize=lambda data: ArbitraryObject(code=data["code"]),
    )

    yield

    DESERIALIZERS.pop()


def test_registration(register_arbitrary_object) -> None:
    instance = deserialize({"code": "test"})

    assert isinstance(instance, ArbitraryObject)
    assert instance.code == "test"


def test_registeration_mixup() -> None:
    data_that_looks_like_prior = {
        "dist": "Normal",
        "kwargs": {"something": "else"},
    }

    match = "Couldn't deserialize"
    with pytest.raises(DeserializableError, match=match):
        deserialize(data_that_looks_like_prior)
