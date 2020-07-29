
from model.input_fn import *
from model.model_fn import *
from model.training import *
from model.utils.utils import Params

if __debug__:
    print('DEBUG')

    data_path = 'dataset/test/' 
    if os.path.isfile(data_path):
        print('%s does not exists '% data_path)       
    # set the data path 
    image_path = data_path + 'image/'
    force_path = data_path + 'forces/force.txt'

    # load the params from json file
    json_path = os.path.join('experiments/', 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    images_list = glob.glob(image_path + '*.jpg')
    force_list = load_force_txt(force_path,len(images_list))
    image_size = (224, 224 ,3)

    print('*****************************')
    print('Dataset is built by %d images'% len(images_list))


    # input_fn(True, images_list, force_list, params= params)
    mode = 'train'
    model_spec = model_fn(mode, params)    
    print('*****************************')

    log_dir = './logs/'

    training_and_eval(model_spec, log_dir, params)
