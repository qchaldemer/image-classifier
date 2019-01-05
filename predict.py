# load module
import json
import predict_utils
from predict_utils import process_image, load_checkpoint, predict
import argparse

parser = argparse.ArgumentParser(
    description='Parameters for predict')
parser.add_argument('--input', action="store",
                    dest="input", default = 'checkpoint4.pth')
parser.add_argument('--top_k', action="store",
                    dest="top_k", default = '5')
parser.add_argument('--image', action="store",
                    dest="image", default = 'flowers/test/100/image_07899.jpg')
parser.add_argument('--device', action="store",
                    dest="device", default = 'gpu')
parser.add_argument('--category_names', action="store",
                    dest="category_names", default = 'cat_to_name')

args = vars(parser.parse_args())

#imputs
image_path = args['image']
checkpoint = args['input']
topk = int(args['top_k'])
device = 'cuda' if args['device'] == 'gpu' else 'cpu'

# load the model
model, learning_rate, hidden_units, class_to_idx = load_checkpoint(checkpoint)

# prediction
probs, classes = predict(image_path, device, model, topk)

# print results
cat_to_name = args['category_names'] + '.json'
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
top_labels = [cat_to_name[cat] for cat in classes]

res = "\n".join("{} {}".format(x, y) for x, y in zip(probs, top_labels))

print(res)

# 



