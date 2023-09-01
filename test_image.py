import argparse
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator  # Make sure to import your Generator class from the model module

parser = argparse.ArgumentParser(description="Test Single Image")
parser.add_argument('--upscale_factor', default=4, type=int, help="super resolution upscale factor")
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low-resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()

# Load the model weights without specifying map_location
model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=torch.device('cpu')))


image = Image.open(IMAGE_NAME).convert('RGB')
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)

if TEST_MODE:
    image = image.cuda()  # This line is safe to keep if you plan to use GPU in the future

start = time.time()
out = model(image)
elapsed = (time.time() - start)
print('cost ' + str(elapsed) + 's')

out_image = ToPILImage()(out[0].data.cpu())
out_image.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
