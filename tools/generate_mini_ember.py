import numpy as np
import jsonlines as jsonl


def encode_label(j):
    e = np.zeros((2,))
    e[j] = 1.0
    return e


def load_jsonl(path):
    with jsonl.open(path) as reader:
        count = 0
        features, labels = [], []
        for obj in reader:
            if count >= 55:
                break
            features.append(obj['byteentropy'])
            labels.append(obj['label'])
            count += 1
    return features, labels


def load_ember():
    benign_path = 'ember_2017_2/train_features_0.jsonl'
    malware_path = 'ember_2017_2/train_features_3.jsonl'
    benign_features, benign_labels = load_jsonl(benign_path)
    malware_features, malware_labels = load_jsonl(malware_path)
    x_train = np.array(benign_features[:50] + malware_features[:50])
    y_train = np.array([encode_label(i) for i in benign_labels[:50] + malware_labels[:50]])
    x_test = np.array(benign_features[50:] + malware_features[50:])
    y_test = np.array([encode_label(i) for i in benign_labels[50:] + malware_labels[50:]])
    max_x_value = np.max(np.concatenate([x_train, x_test]))
    return (x_train / max_x_value, y_train), (x_test / max_x_value, y_test)


def transform_ember_to_mini_ember(data, name, size):
    mini_data = [(x, y) for x, y in zip(data[0], data[1])]
    with open(f'x_{name}_ember', 'w') as f, open(f'y_{name}_ember', 'w') as g:
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


def get_mini_ember():
    train, test = load_ember()
    transform_ember_to_mini_ember(train, 'train', len(train[0]))
    transform_ember_to_mini_ember(test, 'test', len(test[0]))


if __name__ == "__main__":
    get_mini_ember()
