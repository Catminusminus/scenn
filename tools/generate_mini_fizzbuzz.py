import numpy as np


def encode_label(index):
    e = np.zeros((4,))
    e[index] = 1.0
    return e


def encode_fizzbuzz(num):
    if num % 15 == 0:
        return encode_label(0)
    if num % 5 == 0:
        return encode_label(1)
    if num % 3 == 0:
        return encode_label(2)
    return encode_label(3)


def preprocess_num_to_array(num):
    return np.array([num % i for i in range(1,20)])


def load_fizzbuzz():
    x_train = np.random.randint(10000, size=1000)
    y_train = np.frompyfunc(encode_fizzbuzz, 1, 1)(x_train)
    x_test = np.random.randint(10000, size=30)
    y_test = np.frompyfunc(encode_fizzbuzz, 1, 1)(x_test)
    original_x_train = np.copy(x_train)
    original_x_test = np.copy(x_test)
    x_train = np.frompyfunc(preprocess_num_to_array, 1, 1)(x_train)
    x_test = np.frompyfunc(preprocess_num_to_array, 1, 1)(x_test)

    return (x_train, y_train), (x_test, y_test), (original_x_train, original_x_test)


def write_mini_data(data, name, size):
    mini_data = [(x, y) for x, y in zip(data[0], data[1])]
    with open(f'x_{name}_fizzbuzz', 'w') as f, open(f'y_{name}_fizzbuzz', 'w') as g:
        for i, (x, y) in enumerate(mini_data[:size]):
            f.write('{')
            for j_x, x_e in enumerate(x):
                f.write(str(x_e))
                if j_x + 1 != len(x):
                    f.write(',')
            f.write('}')
            if i + 1 != size:
                f.write(',')
            g.write('{')
            for j_y, y_e in enumerate(y):
                g.write(str(y_e))
                if j_y + 1 != len(y):
                    g.write(',')
            g.write('}')
            if i + 1 != size:
                g.write(',')


def write_orig_data(data, name):
    with open(f'{name}_fizzbuzz_orig', 'w') as f:
        for j_X, x_e in enumerate(data):
            f.write(str(x_e))
            if j_X + 1 != len(data):
                f.write(',')


def get_mini_fizzbuzz():
    train, test, orignal = load_fizzbuzz()
    write_mini_data(train, 'train', len(train[0]))
    write_mini_data(test, 'test', len(test[0]))
    write_orig_data(orignal[0], 'train')
    write_orig_data(orignal[1], 'test')


if __name__ == "__main__":
    get_mini_fizzbuzz()
