import yaml

data = {
    "train" : '/shared_hdd/annasdfghjkl13/APT/data/real_data/train', #절대경로 작성
        "val" : '/shared_hdd/annasdfghjkl13/APT/data/real_data/val', #절대경로 작성
        "names" : {
            0: 'airplane',
            1: 'antelope',
            2: 'bear',
            3: 'bicycle',
            4: 'bird',
            5: 'bus',
            6: 'car',
            7: 'cattle',
            8: 'dog',
            9: 'domestic cat',
            10: 'elephant',
            11: 'fox',
            12: 'giant panda',
            13: 'hamster',
            14: 'horse',
            15: 'lion',
            16: 'lizard',
            17: 'monkey',
            18: 'motorcycle',
            19: 'rabbit',
            20: 'red panda',
            21: 'sheep',
            22: 'snake',
            23: 'squirrel',
            24: 'tiger',
            25: 'train',
            26: 'turtle',
            27: 'watercraft',
            28: 'whale',
            29: 'zebra'
            }} #라벨이랑 라벨명 작성

with open('./real_data.yaml', 'w') as f :
    yaml.dump(data, f)

# check written file
with open('./real_data.yaml', 'r') as f :
    lines = yaml.safe_load(f)
    print(lines)
    
    