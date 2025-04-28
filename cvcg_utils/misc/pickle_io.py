def load_pickle(path):
    import pickle
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def dump_pickle(path, data):
    import pickle
    with open(path, 'wb') as file:
        pickle.dump(data, file)
