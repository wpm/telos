from random import randint
from typing import List


def structured_sequence(n: int = 10, m: int = 9, r: int = 3, l: int = 2) -> List[int]:
    """
    Generate a sequence of n digits from 0 to m with sequential structure.
    """

    def sub_sequence() -> List[int]:
        t = [i]
        for _ in range(l):
            t.append(t[-1] + randint(0, 1))
        return t

    s = []
    i = 2 * randint(0, r)
    for _ in range(n // (l + 1) + 1):
        s.extend(sub_sequence())
    return [i % (m + 1) for i in s][:n]


def random_sequence(n: int = 10, m: int = 9) -> List[int]:
    """
    Generate a random sequence of n digits from 0 to m.
    """
    return [randint(0, m) for _ in range(n)]


def main():
    print('Structured')
    for _ in range(10):
        print(structured_sequence())
    print('Random')
    for _ in range(10):
        print(random_sequence())


if __name__ == '__main__':
    main()
