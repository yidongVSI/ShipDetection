'''
Generate Fast RCNN training labels
THe output will be axises-alignted bounding box
'''
import os
import argparser

from ShipDataset import ShipDataset
import ShipUtils as utils

def arg_parser():
  parser = argparse.ArgumentParser(description="Generate axis-aligned box from mask annotation for Fast RCNN training",
                                   epilog=""
                                   )
  req_parse = parser.add_argument_group('required arguments')
  # required arguments
  req_parse.add_argument('-i',    dest = 'img_dir',  required = True, help = 'input image directory')
  req_parse.add_argument('--csv', dest = 'csv_file', required = True, help = 'annotation csv file')
  req_parse.add_argument('-o',    dest = 'out_dir',  required = True, help = 'output folder to put')
  parser.add_argument('--vis',    dest = 'is_vis', default = False, action='store_true',
                        help = 'option to generate visualization.  Default is False')
  return parser.parse_args()

def generate_frcnn_training_label(img_dir, csv_file, out_dir, is_vis = False):
  