import pytest

from asrom_llm.utils import (
    call_api,
    chunks,
    get_key_from_value,
    load_json,
    load_pickle,
    save_json,
    save_pickle,
    verbose_print,
)


def test_json_operations(tmp_path):
    test_data = {"key": "value"}
    json_file = tmp_path / "test.json"

    save_json(test_data, json_file)
    loaded_data = load_json(json_file)

    assert test_data == loaded_data


def test_pickle_operations(tmp_path):
    test_data = {"key": "value"}
    pickle_file = tmp_path / "test.pkl"

    save_pickle(test_data, pickle_file)
    loaded_data = load_pickle(pickle_file)

    assert test_data == loaded_data


def test_verbose_print(capsys):
    verbose_print("Test text", True)
    captured = capsys.readouterr()
    assert captured.out == "Test text\n"

    verbose_print("Test text", False)
    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.parametrize(
    "n, lists, expected_output",
    [
        (
            2,
            ([1, 2, 3, 4], [5, 6, 7, 8]),
            [([1, 2], [5, 6]), ([3, 4], [7, 8])],
        ),
        (
            3,
            ([1, 2, 3], [4, 5, 6], [7, 8, 9]),
            [([1, 2, 3], [4, 5, 6], [7, 8, 9])],
        ),
    ],
)
def test_chunks(n, lists, expected_output):
    result = list(chunks(n, *lists))
    assert result == expected_output


def test_call_api():
    def test_func(x):
        return x * 2

    result = call_api(test_func, 3, 1, 5)
    assert result == 10

    with pytest.raises(Exception):

        def failing_func(x):
            raise Exception("Test exception")

        call_api(failing_func, 3, 1, 5)


def test_get_key_from_value():
    test_dict = {"key1": "value1", "key2": "value2"}

    assert get_key_from_value("value1", test_dict) == "key1"
    assert get_key_from_value("value3", test_dict) is None
