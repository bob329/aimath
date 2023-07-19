from functools import reduce
def Factorial(x):
    return reduce(lambda x, y: x * y, range(1, x + 1))
if __name__ == '__main__':
    print(Factorial(3))