import os
import time

import tensorflow as tf
import tensorflow_datasets as tfds

import re


def create_pipeline(dataset: tf.data.Dataset, batch_size: int, normalize_fn: callable,
                    reproducible: bool = True) -> tf.data.Dataset:
    # make double sure all determinism features are enabled
    options = tf.data.Options()
    options.deterministic = True
    #    dataset = dataset.with_options(options)

    dataset = dataset.map(lambda x: (tf.cast(x['image'], tf.float32), x['label'], x['tfds_id']))

    dataset = dataset.shuffle(seed=1337 if reproducible else None, buffer_size=512,
                              reshuffle_each_iteration=True)  # shuffle 512 samples

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y, z: (normalize_fn(x), y, z))  # data normalization

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def print_ex_ids(
        ds: tf.data.Dataset = None,
        take: int = 10,
        info: tfds.core.DatasetInfo = None,
        skip: int = None,
) -> None:
    """Print the example ids from the given dataset split."""

    if skip:
        ds = ds.skip(skip)
    ds = ds.take(take)
    exs = [z.numpy() for (x, y, z) in ds]
    ids = []
    for ex in exs:
        ids.extend([s.decode('utf-8') for s in ex])
    exs = [format_id(tfds_id, info=info) for tfds_id in ids]
    print(exs)


def format_id(tfds_id: str, info) -> str:
    """Format the tfds_id in a more human-readable."""
    match = re.match(r'\w+-(\w+).\w+-(\d+)-of-\d+__(\d+)', tfds_id)
    split_name, shard_id, ex_id = match.groups()
    split_info = info.splits[split_name]
    return sum(split_info.shard_lengths[:int(shard_id)]) + int(ex_id)


def seed_everything(seed: int):
    tf.keras.utils.set_random_seed(seed)  # set random seed for keras, numpy, tensorflow, and the 'random' module
    os.environ['PYTHONHASHSEED'] = str(seed)


def normalize_img(image):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.


def load_data(dataset_name: str, reproducible: bool):
    if reproducible:
        read_config = tfds.ReadConfig(
            shuffle_seed=32,
            add_tfds_id=True
        )
    else:
        read_config = tfds.ReadConfig(
            add_tfds_id=True
        )
    # we need the 'with_info' to get the split info, which we use to deduce the example ide
    train_ds, info = tfds.load(dataset_name, split='train',
                               shuffle_files=True, read_config=read_config,
                               with_info=True)

    train_ds = create_pipeline(train_ds, batch_size=4, normalize_fn=normalize_img, reproducible=reproducible)

    return train_ds, info


def run_pipeline(args, reproducible=False):
    start = time.time()
    dataset, info = load_data(dataset_name=args.dataset_name, reproducible=reproducible)
    print("First call:")
    print_ex_ids(ds=dataset, take=5, info=info, skip=0)

    print(
        f'Loaded {"non-reproducible" if not reproducible else "reproducible"} dataset in {time.time() - start:.2f} seconds')

    start = time.time()
    dataset, info = load_data(dataset_name=args.dataset_name, reproducible=reproducible)
    print("Second call:")
    print_ex_ids(ds=dataset, take=5, info=info, skip=0)
    print(
        f'Loaded {"non-reproducible" if not reproducible else "reproducible"} dataset in {time.time() - start:.2f} seconds')


def run_pipeline_reproducible(args):
    seed_everything(1337)
    run_pipeline(args, reproducible=True)


if __name__ == "__main__":
    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--dataset-name', type=str, default="mnist",
                                 help="Name of the dataset to use, must map a name from TFDS, e.g. cifar100 or mnist")
    args = argument_parser.parse_args()

    seed_everything(1337)
    run_pipeline(args)
    run_pipeline_reproducible(args)
