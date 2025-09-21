import torch
import cv2
import argparse
import numpy as np
from PIL import Image
from ml_framework.models.UNet import UNet
from ml_framework.framework.SupervisedMLFramework import SupervisedMLFramework


def main(args):

    image = torch.from_numpy(np.array(Image.open(args.sample_path), dtype=np.float32)[:,:,:3])
    image = torch.permute(image, (2,0,1))
    image = torch.unsqueeze(image, dim=0)

    model = UNet(in_ch=3)
    model.load_state_dict(torch.load(args.weights_path, weights_only=True))
    output_path = "."

    framework = SupervisedMLFramework(model, "UNet", output_path)
    prediction = framework.predict(image)

    # get rid of batch axis
    prediction = torch.squeeze(prediction)
    output_mask = torch.permute(prediction, (1,2,0))

    output_mask = torch.argmax(prediction, 0).numpy()
    output_mask[output_mask > 0] = 255

    cv2.imwrite(f"{args.output_path}/eval_output_img.png", output_mask.astype(np.uint8))
    cv2.imwrite(f"{args.output_path}/eval_input_img.png", np.array(Image.open(args.sample_path)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sample-path", help="path to sample", required=True)
    parser.add_argument("-w", "--weights-path", help="path to weights", required=True)
    parser.add_argument("-o", "--output-path", help="path to output", required=True)
    args = parser.parse_args()

    main(args)

