data = [
    {'a': 2, 'b': 3},
    {'a': 1, 'b': 1},
    {'a': 2, 'b': 1},
    {'a': 1, 'b': 2}
]

data = sorted(data, key=lambda d: (d['b'], -d['a']))
print(data)