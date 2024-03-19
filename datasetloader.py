import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image


class KenyanFood13Dataset(Dataset):

    def __init__(self, data_root, train=True, image_shape=None, transform=None):
        """
        init method of the class.
         Parameters:
         data_root (string): path of root directory.
         train (boolean): True for training dataset and False for test dataset.
         image_shape (int or tuple or list): [optional] int or tuple or list. Defaut is None.
                                             If it is not None image will resize to the given shape.
         transform (method): method that will take PIL image and transform it.
        """

        # get label to species mapping
        train_csv_path = os.path.join(data_root, "train.csv")

        # random.seed(21)

        pd_csv = pd.read_csv(train_csv_path)

        # set image_resize attribute
        if image_shape is not None:
            if isinstance(image_shape, int):
                self.image_shape = (image_shape, image_shape)

            elif isinstance(image_shape, tuple) or isinstance(image_shape, list):
                assert (
                    len(image_shape) == 1 or len(image_shape) == 2
                ), "Invalid image_shape tuple size"
                if len(image_shape) == 1:
                    self.image_shape = (image_shape[0], image_shape[0])
                else:
                    self.image_shape = image_shape
            else:
                raise NotImplementedError

        else:
            self.image_shape = image_shape

        # set transform attribute
        self.transform = transform

        unique_classes = pd_csv["class"].unique()

        num_classes = len(unique_classes)

        classes_pd = pd_csv["class"]
        image_paths = pd_csv["id"]
        self.classes, counter = {}, 0
        for uclass in unique_classes:
            self.classes[uclass] = counter
            counter += 1

        # initialize the data dictionary
        self.data_dict = {"image_path": [], "label": []}

        image_classes = [[] for _ in range(len(unique_classes))]
        for ind, path in enumerate(image_paths):
            image_classes[self.classes[classes_pd[ind]]].append(path)

        im_path = os.path.join(data_root, "images", "images")

        for c, image_class in enumerate(image_classes):
            X_train, X_test, y_train, y_test = train_test_split(
                image_class, [c] * len(image_class), test_size=0.20, random_state=23
            )
            if train:
                X, y = X_train, y_train
            else:
                X, y = X_test, y_test
            for im in X:
                self.data_dict["image_path"].append(
                    os.path.join(im_path, str(im) + ".jpg")
                )
                self.data_dict["label"].append(c)

    def __len__(self):
        """
        return length of the dataset
        """
        return len(self.data_dict["label"])

    def __getitem__(self, idx):
        """
        For given index, return images with resize and preprocessing.
        """

        image = Image.open(self.data_dict["image_path"][idx]).convert("RGB")

        # if self.image_shape is not None:
        #     image = F.resize(image, self.image_shape)

        if self.transform is not None:
            image = self.transform(image)

        target = self.data_dict["label"][idx]

        return image, target
