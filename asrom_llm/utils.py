import json
import pickle


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def verbose_print(text, verbose, **kwargs):
    if verbose:
        print(text, **kwargs)


def chunks(n=5000, *lists):
    """Yield successive n-sized chunks as tuples from multiple lists."""
    max_length = max(len(lst) for lst in lists)

    for i in range(0, max_length, n):
        chunk_tuple = tuple(lst[i : i + n] for lst in lists)
        yield chunk_tuple
