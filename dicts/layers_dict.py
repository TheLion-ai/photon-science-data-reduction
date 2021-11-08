

layers_dict = {
    'lenet': [
        'Conv 1', 'Tanh 1', 'AvgPool 1',
        'Conv 2', 'Tanh 2', 'AvgPool 2',
        'Conv 3', 'Tanh 3', 'AvgPool 3'
    ],

    'alexnet': [
        'Conv 1', 'ReLU 1', 'MaxPool 1',
        'Conv 2', 'ReLU 2', 'MaxPool 2',
        'Conv 3', 'ReLU 3',
        'Conv 4', 'ReLU 4',
        'Conv 5', 'ReLU 5', 'MaxPool 5'
    ],

    'vgg19' : [
        'Conv 1', 'ReLU 1',
        'Conv 2', 'ReLU 2', 'MaxPool 2',
        'Conv 3', 'ReLU 3',
        'Conv 4', 'ReLU 4', 'MaxPool 4',
        'Conv 5', 'ReLU 5',
        'Conv 6', 'ReLU 6',
        'Conv 7', 'ReLU 7',
        'Conv 8', 'ReLU 8', 'MaxPool8',
        'Conv 9', 'ReLU 9',
        'Conv 10', 'ReLU 10',
        'Conv 11', 'ReLU 11',
        'Conv 12', 'ReLU 12', 'MaxPool 12',
        'Conv 13', 'ReLU 13',
        'Conv 14', 'ReLU 14',
        'Conv 15', 'ReLU 15',
        'Conv 16', 'ReLU 16', 'MaxPool 16'
    ]
}