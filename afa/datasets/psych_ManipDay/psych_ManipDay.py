import os
import pickle

import gdown
import tensorflow as tf
import tensorflow_datasets as tfds

url="https://drive.google.com/file/d/1icIFsOIZgktWHe4W9PPmwI7_Ovf0zUn5/view?usp=sharing"

class Psych_ManipDay(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="Psychology Dataset with ManipDay as Target Label",
            features=tfds.features.FeaturesDict(
                {
                    "features": tfds.features.Tensor(shape=(40,), dtype=tf.float32),
                    "label": tfds.features.ClassLabel(num_classes=2),
                }
            ),
            supervised_keys=("features", "label"),
            citation=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        data_path = gdown.download(
            url=url, output=os.path.join(dl_manager.download_dir, "psych_ManipDay.pkl"), fuzzy=True 
        )

        with open(data_path, "rb") as fp:
            data = pickle.load(fp)

        return {
            "train": self._generate_examples(data["train"]),
            "validation": self._generate_examples(data["valid"]),
            "test": self._generate_examples(data["test"]),
        }

    def _generate_examples(self, data):
        for i, (x, y) in enumerate(zip(*data)):
            yield i, dict(features=x, label=y)
