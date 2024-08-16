import argparse
import os
import matplotlib.pyplot as plt
from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', type=str, default=None, help='Input image or directory of images')
parser.add_argument('--use_gpu', action='store_true', help='Whether to use GPU')
parser.add_argument('-o', '--save_prefix', type=str, default=None, help='Output directory')
opt = parser.parse_args()

# Load colorizer
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if opt.use_gpu:
    colorizer_siggraph17.cuda()

# Check if the input path is a directory or a file
if os.path.isdir(opt.img_path):
    img_files = [os.path.join(opt.img_path, f) for f in os.listdir(opt.img_path) if f.endswith(('jpg', 'jpeg', 'png'))]
else:
    img_files = [opt.img_path]

# Create output directory if it doesn't exist
if not os.path.exists(opt.save_prefix):
    os.makedirs(opt.save_prefix)

for img_file in img_files:
    try:
        # Load image
        img = load_img(img_file)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        if opt.use_gpu:
            tens_l_rs = tens_l_rs.cuda()

        # Colorization using SIGGRAPH 17 model
        out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

        # Save results
        base_name = os.path.basename(img_file).split('.')[0]
        output_path = os.path.join(opt.save_prefix, f'{base_name}')
        plt.imsave(output_path, out_img_siggraph17)

        print(f'Processed and saved: {output_path}')

    except Exception as e:
        print(f'Error processing file {img_file}: {e}')
