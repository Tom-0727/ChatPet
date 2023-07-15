import fire
import torch

def main(a: str = '1',
         b: str = '2',
         c: str = '3'):
    print(a, b, c)

    x = torch.tensor([1, 2, 3, 4, 5])
    y = torch.tensor([777])
    condition = torch.tensor([True, True, True, True, False])
    result = torch.where(condition, x, y)
    print(result)


if __name__ == '__main__':
    fire.Fire(main)
