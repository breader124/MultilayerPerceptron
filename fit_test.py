from matplotlib import pyplot as plt
import numpy as np
from random import seed
import seaborn as sns

sns.set()

from neural_network import NeuralNetwork

if __name__ == '__main__':
    seed(42)
    np.random.seed(42)

    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    model = NeuralNetwork(2, [2], 1)
    in_ = X[0, :]
    exp = y[0]
    out = model.predict(in_)
    print(out)

    errs = model.batch_fit(X, y, batch=4, reps=3000, beta=0.5)
    plt.plot(errs)
    plt.legend(['Error'])
    plt.show()

    for x in X:
        out = model.predict(x)
        print(f'{x}: {out}')
