import json


def write_json(data):
    j = json.dumps(data)
    with open('../data/db.json', 'w') as f:
        f.write(j)


def read_json():
    try:
        with open('../data/db.json', 'r') as f:
            return json.load(f)
    except:
        return {}
