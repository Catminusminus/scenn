import numpy as np


def load_mnist():
    path = 'mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def transform_mnist_to_mini_mnist(data, name):
    mini_data = [(x, y) for x, y in zip(data[0], data[1]) if y == 0 or y == 1 or y == 2]
    with open(f'x_{name}', 'w') as f, open(f'y_{name}', 'w') as g:
        for (x, y) in mini_data:
            f.write('{')
            for x_e in x:
                f.write(x_e)
                f.write(',')
            f.write('}')
            f.write(',')
            g.write(y)
            g.write(',')


def get_mini_mnist():
    train = load_mnist()
    transform_mnist_to_mini_mnist(train, 'train')
    test = load_mnist()
    transform_mnist_to_mini_mnist(test, 'test')
