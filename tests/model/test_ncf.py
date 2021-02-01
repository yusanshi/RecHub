import torch
from rechub.parameters import parse_args
from rechub.model.ncf import NCF
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

args = parse_args()
args.model_name = 'NCF'
model = NCF(args, 100, 100).to(device)
user_index = torch.randint(100, (64, )).to(device)
item_index = torch.randint(100, (64, )).to(device)
first = {'name': 'user', 'index': user_index}
second = {'name': 'item', 'index': item_index}
y_pred = model(first, second)
print(y_pred)
