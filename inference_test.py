"""
inference test
"""

import argparse
import os
from packaging import version
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model.input_fn import *
from model.model_fn import *
from model.training import *

from tensorflow.keras.preprocessing import image

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./log/20200825-143805/',
                    help="Experiment directory containing params.json")

parser.add_argument('--data_dir', default='./data_single',
                    help="Directory containing the dataset")

if __name__ == '__main__':
    


    args = parser.parse_args()

    # check if the model directory is available
    assert os.path.exists(args.model_dir), "No model file found at {}".format(args.model_dir)
    model_path = os.path.join(args.model_dir, 'best_full_model_path')

    test_data_dir = os.path.join(args.data_dir, 'test')
    # Get the filenames from the train and dev sets
    test_filenames = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir)]
    # Get the train images list
    images_list_test = glob.glob(test_filenames[0] + '/*.jpg')
    # Get the label forces 
    force_list_test = load_force_txt(test_filenames[1]+ '/force.txt',len(images_list_test))

    # Create the two iterators over the two datasets
    print('=================================================')

    loaded_model = tf.saved_model.load(model_path)
    print('[INFO] Model loaded...')
    image_path = images_list_test[0]
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    testStart = time.time()
    test_predictions = loaded_model(x)
    testEnd = time.time()
    elapsed = (testEnd - testStart)
    print("[INFO] 1 inference Took {:.4} Seconds".format(elapsed))
    print('=================================================')