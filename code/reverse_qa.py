import json

def read_data(path):
    data = json.load(open(path))
    return data

def save_data(data, name):
    with open(name,"w") as f:
        json.dump(data, f)
    
def reverse(data):
    for dialog in data["dialogs"]:
        print("Processing image: %s"%dialog["image_id"], end="")
        for i in range(len(dialog["dialog"])):
            print(".", end="")
            dialog["dialog"][i]["question"], dialog["dialog"][i]["answer"] = dialog["dialog"][i]["answer"], dialog["dialog"][i]["question"]
        print("")

    return data

if __name__=="__main__":
    PATH = "/Users/tslab/Desktop/train_set4DSTC7-AVSD.json"
    NAME = "train_reversedQA.json"
    data = read_data(PATH)
    data = reverse(data)
    save_data(data,NAME)

