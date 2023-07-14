import fire


def main(a: str = '1',
         b: str = '2',
         c: str = '3'):
    print(a, b, c)


if __name__ == '__main__':
    fire.Fire(main)
