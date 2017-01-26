from argparse import ArgumentParser
from StylingModel import StylingModel

parser = ArgumentParser()

# Inputs and outputs
parser.add_argument("content_path", type=str, help="Path to the content image")
parser.add_argument("style_path", type=str, help="Path to the style image")
parser.add_argument("-tp", "--target_path", type=str, default="./")

# Loss weights
parser.add_argument("-cw", "--content_weight", type=float, default=0.1)
parser.add_argument("-sw", "--style_weight", type=float, default=5000000.0)
parser.add_argument("-tvw", "--tv_weight", type=float, default=0.01)

# Number of iterations and image generation frequency
parser.add_argument("-it", "--iterations", type=int, default=1000)
parser.add_argument("-f", "--frequency", type=int, default=10)

args = parser.parse_args()

style_transfer = StylingModel(content_path=args.content_path, style_path=args.style_path)

style_transfer.optimize(args.content_weight, args.style_weight, args.tv_weight)

style_transfer.train(args.iterations, args.frequency, args.target_path)
