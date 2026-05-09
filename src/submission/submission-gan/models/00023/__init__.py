import logging
import os
from pathlib import Path


from collections import OrderedDict

from torch.autograd import Variable
import torch

#from .src.options.test_options import TestOptions
from .src.prepost_data.data_loader import CreateDataLoader
from .src.prepost_model.model import create_model
from .src.prepost_util import util as util

import sys

model_root_path = str(Path(os.path.abspath(__file__)).parent)
sys.path.append(model_root_path)
sys.path.append(model_root_path + '/src') 
sys.path.append(model_root_path + '/src/prepost_data') 
sys.path.append(model_root_path + '/src/prepost_model') 
sys.path.append(model_root_path + '/src/prepost_util') 

#root_parent_path = str(Path(model_root_path).parent)
#sys.path.append(root_parent_path) # parent of parent
#sys.path.append(root_parent_path + '/src') # parent of parent
#sys.path.append(root_parent_path + '/src/prepost_data') # parent of parent
#sys.path.append(root_parent_path + '/src/prepost_model') # parent of parent
#sys.path.append(root_parent_path + '/src/prepost_util') # parent of parent


class TestOptions:
    def __init__(self, image_height, checkpoints_dir, checkpoint_name, gpu_ids, num_samples, input_image_dir = 'input'):

        # self.gpu_ids=-1
        if gpu_ids is None or gpu_ids == -1:
            self.gpu_ids = []
        else:
            self.gpu_ids = [gpu_ids]

        ## existing arguments from pix2pix (non-HD) template
        self.name = checkpoint_name #'pre2postcontrast_512p'
        self.checkpoints_dir = checkpoints_dir
        self.verbose = False
        self.which_epoch = '30'
        self.isTrain = False
        self.use_encoded_image = False
        self.load_features = False
        self.instance_feat = False
        self.label_feat = False


        ### pix2pixHD arguments from testOptions
        self.nThreads = 0 # test code only supports nThreads = 1
        self.batchSize = 1 # test code only supports batchSize = 1
        self.serial_batches = True # no shuffle
        self.no_flip = True # no flip
        self.aspect_ratio = 1.0
        self.phase = 'test'

        ### pix2pixHD arguments from baseOptions
        self.feat_num = 3
        self.n_downsample_E = 4
        self.nef = 16
        self.n_clusters = 1
        self.netG='global'
        self.ngf=64
        self.n_downsample_global=4
        self.n_blocks_global=9
        self.n_blocks_local=3
        self.n_local_enhancers=1
        self.niter_fix_global=0   

        self.model='pix2pixHD'
        self.norm='instance'
        self.data_type=32
        self.local_rank=0
        self.how_many = num_samples
        self.image_height = image_height

        # input/output sizes       
        self.loadSize=512
        self.fineSize=512
        self.label_nc=0
        self.input_nc=3
        self.output_nc=3
        self.no_instance = True
        self.tf_log = False

        # for setting inputs
        self.dataroot=input_image_dir
        self.resize_or_crop='scale_width'
        self.max_dataset_size=float("inf")



def test(opt, model, data_loader, dataset, output_path, save_images, num_samples):

    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        logging.info(model)
    image_list = []
    names_list = []
    for i, data in enumerate(dataset):
        if i >= num_samples:
            break
        if opt.data_type == 16:
            data['label'] = data['label'].half()
            data['inst']  = data['inst'].half()
        elif opt.data_type == 8:
            data['label'] = data['label'].uint8()
            data['inst']  = data['inst'].uint8()

        minibatch = 1 
          
        generated = model.inference(data['label'], data['inst'], data['image'])

        img_path = data['path']
        logging.debug('process image... %s' % img_path)
        short_path = os.path.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        #if i < 10:
            #image_list.append(util.tensor2label(data['label'][0], opt.label_nc))
        #else:
            #image_list.append(util.tensor2im(generated.data[0]))
        image_list.append(util.tensor2im(generated.data[0]))
        names_list.append(name)
    if save_images:
        #visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
        #                       ('synthesized_image', util.tensor2im(generated.data[0]))])
        save_generated_images(image_list=image_list, img_path=img_path, output_dir=output_path, names_list=names_list)
    else:
        #return OrderedDict([('synthesized_image', util.tensor2im(generated.data[0]))])
        return_list = []
        for idx, name in enumerate(names_list):
            # per medigan standard, a list of dicts is returned to user.
            return_list.append((image_list[idx], name))

        return return_list



def init_pix2pixHD(image_height, checkpoints_dir, checkpoint_name, gpu_ids, num_samples, input_image_dir):
    try:
        opt = TestOptions(image_height, checkpoints_dir, checkpoint_name, gpu_ids, num_samples, input_image_dir)
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        # test
        model = create_model(opt)
        return opt, model, data_loader, dataset
    except Exception as e:
        logging.error(f"Error while trying to initialize pix2pix: {e}")
        raise e
    

def save_generated_images(image_list, img_path, output_dir, names_list):
    logging.debug(f"output_filepath: {output_dir}")
    try:
        for idx, image_numpy in enumerate(image_list):
            image_name = '%s_syn_%s.jpg' % (names_list[idx], idx)
            save_path = os.path.join(output_dir, image_name)
            util.save_image(image_numpy, save_path)
        logging.info(f"Saved all synthetic images to {output_dir}")
    except Exception as e:
        logging.error(
            f"Error while trying to save generated images in {output_dir}: {e}")
        raise e


def generate(model_file, image_size, input_path, num_samples, save_images, output_path, gpu_id):

    try:
        # instantiate the model
        logging.debug("Instantiating model...")
        checkpoints_dir, checkpoint_name = os.path.split(model_file)
        #rescale_height = image_size[0]  # 512
        #rescale_width = image_size[1]  # 512
        #image_size = (rescale_height, rescale_width)
        device = torch.device("cuda" if (torch.cuda.is_available() and gpu_id is not None and gpu_id != -1) else "cpu")
        if str(device) == 'cuda':
            gpu_ids = gpu_id
        else:
            gpu_ids = None  # 'cpu'


        logging.debug(
            f"checkpoints_dir:{checkpoints_dir}, "
            f"checkpoint_name: {checkpoint_name}, "
            f"gpu_ids: {gpu_ids}, "
            f"input_path: {input_path}, "
            f"output_path: {output_path}, "
            f"image_size: {image_size}, "
            f"num_samples: {num_samples}, "
            f"save_images: {save_images}, ")

        if input_path is None or not os.path.isdir(input_path):
            #input_path = 'input/'
            logging.info(f"Input path {input_path} is not valid. Using {Path(Path(__file__).parent.resolve() / 'input')} as fallback.")
            input_path = Path(Path(__file__).parent.resolve() / "input")

        
        # Rescaling of the image inside the method
        #if os.path.isdir(input_path):
        #    # if a folder is given, the images in that folder are loaded to be a sample_pool 
        #    image_paths = sorted([os.path.join(input_path, file) for file in os.listdir(input_path) if
        #                          "img" in file and file.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])

        opt, model, data_loader, dataset = init_pix2pixHD(image_size, checkpoints_dir, checkpoint_name, gpu_ids, num_samples, input_image_dir = input_path)
        
        logging.debug("Generating images...")

        if output_path is None and save_images:
            logging.warn(f"Output path {output_path} is None and therefore not valid. Using {Path(Path(__file__).parent.resolve() / 'output')} as fallback.")
            output_path = Path(Path(__file__).parent.resolve() / 'output')
        if save_images:
            Path(output_path).mkdir(parents=True, exist_ok=True)

        return test(opt, model, data_loader, dataset, output_path, save_images, num_samples)


    except Exception as e:
        logging.error(f"Error while trying to generate {num_samples} images with model {model_file}: {e}")
        raise e
