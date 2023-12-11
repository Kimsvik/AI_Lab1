def p1_read(path, encoding):
    data_file = pd.read_csv(path, encoding=encoding)
    return data_file