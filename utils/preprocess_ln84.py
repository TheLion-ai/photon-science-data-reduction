import os
from sklearn.model_selection import train_test_split

def get_train_val_split():
    labels = []
    img_names = []
    img_names2 = []

    with open('datasets/ln84/labels.txt', 'r') as f:
        for line in f:
            img_name = line[line.find("[")+1:line.find("]")]
            img_name = img_name.zfill(5)
            img_name = ''.join([img_name, '.png'])
            if line.find('MISS')!=-1:
                label = '0'
            elif line.find('HIT')!=-1:
                label = '1'
            else:
                img_names2.append(img_name)
                continue
            img_names.append(img_name)
            labels.append(label)

    train, val = train_test_split(img_names, train_size=0.8, shuffle=True, stratify=labels)
    
    for idx, img_name in enumerate(train):
        label = labels[idx]
        os.replace(f'datasets/ln84_raw/{img_name}', f'datasets/ln84_raw/train/{label}/{img_name}')
        os.replace(f'datasets/ln84_preprocessed/{img_name}', f'datasets/ln84_preprocessed/train/{label}/{img_name}')

    for idx, img_name in enumerate(val):
        label = labels[idx]
        os.replace(f'datasets/ln84_raw/{img_name}', f'datasets/ln84_raw/val/{label}/{img_name}')
        os.replace(f'datasets/ln84_preprocessed/{img_name}', f'datasets/ln84_preprocessed/val/{label}/{img_name}')

    for img_name in img_names2:
        os.replace(f'datasets/ln84_raw/{img_name}', f'datasets/ln84_raw/2/{img_name}')
        os.replace(f'datasets/ln84_preprocessed/{img_name}', f'datasets/ln84_preprocessed/2/{img_name}')

if __name__ == '__main__':
    get_train_val_split()