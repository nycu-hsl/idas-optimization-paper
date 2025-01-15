import random
from math import ceil

import numpy as np
import parameters as p


def generate(total, architectures, available_architectures=list(i + 1 for i in range(10))):
    remaining = total
    result = []
    for a in architectures[:-1]:
        r = round(random.uniform(0, remaining), 3)
        remaining -= r
        result.append(r)
    result.append(round(remaining, 3))
    random.shuffle(result)

    # pad with 0s on remaining positions
    d = {a: v for (a, v) in zip(architectures, result)}
    return [d.get(a, 0) for a in available_architectures]


# def generate_with_lower_bounds(total, lower_bounds: dict) -> dict:
#     remaining = total
#     result = {}
#     keys = list(lower_bounds.keys())
#     random.shuffle(keys)

#     for k in keys[:-1]:
#         # lower_bound = lower_bounds[k]
#         lower_bound = lower_bounds[k]
#         upper_bound = total - sum(result.get(k_, lower_bounds[k_]) for k_ in lower_bounds.keys() if not k_ == k)
#         if upper_bound <= lower_bound:
#             upper_bound = 10000
#             lower_bound = 100
            
#         r = random.randint(lower_bound, upper_bound)
#         result[k] = r
#         remaining -= r
#         if remaining <=0:
#             remaining = 100

#     result[keys[-1]] = remaining

#     return result

def generate_with_lower_bounds(total, lower_bounds: dict) -> dict:
    remaining = total
    result = {}
    keys = list(lower_bounds.keys())
    random.shuffle(keys)

    for k in keys:
        # print(k)
        lower_bound = lower_bounds[k]
        # print('lower_bound ',lower_bound)
        upper_bound = lower_bounds[k]*p.max_upper
        r = random.randint(lower_bound, upper_bound)
        result[k] = r

    return result


if __name__ == '__main__':
    for _ in range(10):
        result = generate(1, (1, 2, 4, 6))
        print(f'{_}: result {result}, sum {sum(result)}')
    res = np.asarray([generate(1, (1, 2, 4, 6)) for _ in range(100_000)])
    print(res.mean(axis=0))

    lower_bounds = {1: 12,
                    2: 11,
                    4: 15,
                    6: 11}

    for _ in range(10):
        result = generate_with_lower_bounds(100, lower_bounds)

        print(f'run {_}: result {result}, sum {sum(result.values())}')
