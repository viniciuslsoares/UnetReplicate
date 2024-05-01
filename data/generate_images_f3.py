import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tifffile
from mmseg.apis import inference_segmentor, init_segmentor
from glob import glob
import argparse
import warnings


"""
test_path = '/data/f3/images/test/'
config_file = '/experiments/setr_pup_teste/setr_pup_512x512_12k_f3_IN1k.py'
checkpoint_file = '/experiments/setr_pup_teste/latest.pth'
output_path = '/experiments/setr_pup_teste_inference/imagens_teste/'
"""

#parse args test_path, config_file, checkpoint_file output_path
def parse_args():
    parser = argparse.ArgumentParser(description='Inference a segmentor')
    parser.add_argument('--config_file', help='config file path')
    parser.add_argument('--checkpoint_file', help='checkpoint file path')
    parser.add_argument('--output_path', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.config_file is None:
        raise ValueError('Please input config file path')
    if args.checkpoint_file is None:
        raise ValueError('Please input checkpoint file path')
    if args.output_path is None:
        raise ValueError('Please input output path')
    
    test_path = "data/f3/images/test/"
    config_file = args.config_file
    checkpoint_file = args.checkpoint_file
    output_path = args.output_path

    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    test_images = [test_path + 'il_32.tif', test_path + 'il_109.tif', test_path +'il_99.tif']

    #create colormap with this colors: [75, 112, 187],[150, 194, 221],[227, 246, 249],[250, 223, 119],[245, 120, 75],[216, 40, 36]
    cmap = ListedColormap([[0.29411764705882354, 0.4392156862745098, 0.7333333333333333],
                           [0.5882352941176471, 0.7607843137254902, 0.8666666666666667],
                           [0.8901960784313725, 0.9647058823529412, 0.9764705882352941],
                           [0.9803921568627451, 0.8745098039215686, 0.4666666666666667],
                           [0.9607843137254902, 0.47058823529411764, 0.29411764705882354],
                           [0.8470588235294118, 0.1568627450980392, 0.1411764705882353]]
                            )

    for test_fname in test_images:
        img = tifffile.imread(test_fname)
        result = inference_segmentor(model, img)
        file_name = test_fname.split('/')[-1].split('.')[0]
        plt.imsave(output_path + file_name + '.png', result[0], cmap=cmap)

if __name__ == '__main__':
    main()