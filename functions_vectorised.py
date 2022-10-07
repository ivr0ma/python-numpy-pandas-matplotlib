import numpy as np


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    diag = np.diag(X)
    d1 = diag[diag >= 0]
    if d1.shape[0] == 0:
        return -1
    else:
        return np.sum(d1)


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    return np.array_equal(x_sorted, y_sorted)


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    if np.any(x % 3 == 0) and np.size(x) > 2:
        x1 = np.copy(x)
        res = x1[1:] * x[:-1]
        res = res[res % 3 == 0]
        return np.amax(res)
    else:
        return -1


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    return image.dot(weights)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    xn = np.repeat(x[:, 0], x[:, 1])
    yn = np.repeat(y[:, 0], y[:, 1])
    if np.size(xn) != np.size(yn):
        return -1
    else:
        return np.sum(xn * yn)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    X1 = np.sum(X, axis=1)
    X1 = list(np.where(X1 == 0)[0])
    X = np.delete(X, X1, axis=0)

    Y1 = np.sum(Y, axis=1)
    Y1 = list(np.where(Y1 == 0)[0])
    Y = np.delete(Y, Y1, axis=0)

    res = X.dot(Y.T) / np.outer(np.linalg.norm(X, axis=1), np.linalg.norm(Y, axis=1))

    res = np.insert(res, Y1, 1, axis=1)
    res = np.insert(res, X1, 1, axis=0)

    return res
