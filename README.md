# Force Estimation Using Fingernail Imaging with FingerNailNet 

_Author: Navid Fallahinia, University of Utah Robotics Center_

_n.fallahinia@utah.edu_

Take the time to read this [page](http://www.cs.utah.edu/~jmh/Fingernail/index.html) on Fingernail Imaging technique.

## Requirements

We recommend using python3 and a virtual env. When you're done working on the project, deactivate the virtual environment with `deactivate`.

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

Note that this repository uses Tensorflow 2.3.0 and `tf.keras` API. There are major changes between TF 1 and TF 2 and this program does NOT support TF 1. 

Introduction to TF 1 vs. TF 2:
- [programmer's guide](https://www.tensorflow.org/guide/migrate)



## Task

Given an image of a human finger and fingernail apllying force to a surface, predict the 3D contact forces (Fx, Fy, Fz).

## Download the Fingernail dataset

Unfortunately, the Fingernail dataset is not publicly available at this time. However, you can email the [author](n.fallahinia@utah.edu) to receive the datatset. 

Fingernail dataset (~2.6 GB) contains raw and registered images of human fingernails (Index. Middle, Ring and Thumb finger) and corresponding 3D applied forces for 18 human subjects with different fingernail sizes and textures.
Here is the structure of the data:

```
dataset/
    subj_01/
        raw_images/
            img_01_0001.jpg
            ...
        aligned_images/
            image_0001.jpg
            ...
        forces/
            force_01.txt
    subj_02/
        ...
```

The images are named following `image_{IMGIdx}.jpg`. The order of the forces is `[Fx, Fy, Fz]`.

Once the download is complete, move the dataset into the main directory.
Run the script `build_dataset.py` which will split the dataset into train/dev/test subsets and will resize them to (290, 290).

```bash
python build_dataset.py --data_dir data --output_dir dataset
```

## Quick Training

1. **Build the dataset**: make sure you complete this step before training

```bash
python build_dataset.py --data_dir data --output_dir dataset --mode hyper
```

2. **Setup the parameters**: There is a `params.json` file for you under the `experiments` directory. You can set the parameters for the experiment. It looks like

```json
{
    "batch_size": 32,
    "num_epochs": 5,

    "num_channels": 16,
    "predic_layer_size": 1024,
    ...
}
```

For every new training job, you will need to create a new directory under `experiments` with a similar `params.json` file.

3. **Train the model**. Simply run

```
python train.py --data_dir data --model_dir experiments
```

It will instantiate a model and train it on the training set following the parameters specified in `params.json`. It will also evaluate some metrics on the development set.

4. **Hyperparameters search** creat a new directory for the hyperparameter you want to search for in `experiments`. Now, run

```
python search_hyperparams.py --data_dir data --parent_dir experiments/{$HYPER_PARAMETER}
```

It will train and evaluate a model with different values of the spesified hyperparametrs defined in `search_hyperparams.py` and create a new directory for each experiment under `experiments/{$HYPER_PARAMETER}/`.

5. **Evaluation on the test set** Once you've selected the best model and hyperparameters based on the performance on the development set, you can finally evaluate the performance of your model on the test set. Run

```
python test.py --data_dir data --model_dir log/{$BEST_MODEL_PATH}
```
