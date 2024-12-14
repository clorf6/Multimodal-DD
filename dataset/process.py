import json
from collections import defaultdict

data = json.load(open("../data/flickr/flickr30k_train.json", "r"))

merged_data = defaultdict(list)

for item in data:
    image_id = item["image_id"]
    image = item["image"]
    caption = item["caption"]
    if image_id not in merged_data:
        merged_data[image_id] = {"image": image, "captions": []}
    merged_data[image_id]["captions"].append(caption)

result = [
    {"image": data["image"], "captions": data["captions"]}
    for _, data in merged_data.items()
]

with open('../data/flickr/flickr30k_train.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
