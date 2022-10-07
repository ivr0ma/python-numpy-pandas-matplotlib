from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    sum = 0
    flag = False
    m = min(len(X), len(X[0]))
    for i in range(m):
        if X[i][i] >= 0:
            flag = True
            sum += X[i][i]

    if flag:
        return sum
    else:
        return -1


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    x_sorted = sorted(x)
    y_sorted = sorted(y)
    return x_sorted == y_sorted


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    if len(x) <= 2:
        return -1

    max = x[0]*x[1]
    for i in range(len(x)-1):
        c = x[i]*x[i+1]
        if (c % 3 == 0) and (c > max):
            max = c

    if (max % 3 == 0):
        return max
    else:
        return -1


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    num_channels = len(weights)

    height = len(image)
    width = len(image[0])
    res = []
    for i in range(height):
        res.append([0]*width)

    for n in range(num_channels):
        for i in range(height):
            for j in range(width):
                res[i][j] += image[i][j][n] * weights[n]

    return res


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    x_new = []
    for elem in x:
        for i in range(elem[1]):
            x_new.append(elem[0])

    y_new = []
    for elem in y:
        for i in range(elem[1]):
            y_new.append(elem[0])

    if (len(x_new) != len(y_new)):
        return -1

    res = 0
    for i in range(len(x_new)):
        res += x_new[i] * y_new[i]

    return res


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    m = len(X)
    n = len(Y)
    d = len(X[0])
    res = []
    for i in range(m):
        res.append([0] * n)

    for i in range(m):
        for j in range(n):
            p12 = 0
            p1 = 0
            p2 = 0
            for k in range(d):
                p12 += X[i][k] * Y[j][k]
                p1 += X[i][k] ** 2
                p2 += Y[j][k] ** 2

            if (p1 == 0 or p2 == 0):
                res[i][j] = 1
            else:
                res[i][j] = p12 / ( (p1 ** 0.5) * (p2 ** 0.5) )

    return res
