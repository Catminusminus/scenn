import numpy as np


def load_mnist():
    path = 'mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


def encode_label(j):
    e = np.zeros((3,))
    e[j] = 1.0
    return e


def transform_mnist_to_mini_mnist(data, name, size):
    mini_data = [(np.reshape(x, (784,)) / 255, encode_label(y)) for x, y in zip(data[0], data[1]) if y == 0 or y == 1 or y == 2]
    with open(f'x_{name}', 'w') as f, open(f'y_{name}', 'w') as g:
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


def get_mini_mnist():
    train, test = load_mnist()
    transform_mnist_to_mini_mnist(train, 'train', 100)
    transform_mnist_to_mini_mnist(test, 'test', 10)


if __name__ == "__main__":
    get_mini_mnist()
