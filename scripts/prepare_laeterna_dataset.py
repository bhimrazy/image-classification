import glob
import os
import random
import shutil
import subprocess
import pandas as pd

# [STEP 1] Clone the repo
# https://github.com/Z-Unlocked/Unlocked_Challenge_4.git
subprocess.check_call(
    ["git", "clone", "https://github.com/Z-Unlocked/Unlocked_Challenge_4.git"]
)


# clean data/flowers folder if already exists
if os.path.exists("data/flowers"):
    shutil.rmtree("data/flowers/")

# [STEP 2] Prepare datasets folder for train and validation
os.makedirs("data/flowers/train/la_eterna", exist_ok=True)
os.makedirs("data/flowers/train/others", exist_ok=True)
os.makedirs("data/flowers/val/la_eterna", exist_ok=True)
os.makedirs("data/flowers/val/others", exist_ok=True)

# [STEP 3] Move the downloaded data/flowers to train and validation in 80/20 ratio

dir = "Unlocked_Challenge_4/data_cleaned/"  # cloned repo
train_dir = dir + "Train"  # train dir inside cloned repo
test_dir = dir + "scraped_images/image_files/"  # test dir inside cloned repo

# List of Train Images
la_eterna = glob.glob(train_dir + "/la_eterna/*")
others = glob.glob(train_dir + "/other_flowers/*")

val_size = int(len(la_eterna) * 0.2), int(len(others) * 0.2)
random.shuffle(la_eterna)  # shuffling
random.shuffle(others)  # shuffling


def move_images_to_datasets(files, val_size, folder):
    """This function copies the files to the datasets folder.

    Args:
        files : List of all the images inside the given folder
        val_size : Validation size to split the train and val.
        folder : Folder inside train folder
    """
    # Copy images
    destination = "data/flowers/train/" + folder + "/"
    for file in files[val_size:]:
        file_name = file.split("/")[-1]
        dest_path = destination + file_name
        shutil.copy2(file, dest_path)

    destination = "data/flowers/val/" + folder + "/"
    for file in files[:val_size]:
        file_name = file.split("/")[-1]
        dest_path = destination + file_name
        shutil.copy2(file, dest_path)


move_images_to_datasets(
    la_eterna, val_size[0], folder="la_eterna"
)  # move la_eterna images
move_images_to_datasets(others, val_size[1], folder="others")  # move others images

print("Total Datasets: ", len(glob.glob("data/flowers/*/*/*")))
print("Total Train Datasets: ", len(glob.glob("data/flowers/train/*/*")))
print("Total Val Datasets: ", len(glob.glob("data/flowers/val/*/*")))

def create_label(img_path:str):
    """This function creates the label for the image.

    Args:
        img_path (str): Path of the image

    Returns:
        str: label
    """
    if "la_eterna" in img_path:
        return 0 # la_eterna
    else:
        return 1 # others

# create dataframe and csv
train_df = pd.DataFrame(
    glob.glob("data/flowers/train/*/*"), columns=["image_path"]
)
train_df["label"] = train_df["image_path"].apply(create_label)
train_df.to_csv("data/flowers/train.csv", index=False)

test_df = pd.DataFrame(
    glob.glob("data/flowers/val/*/*"), columns=["image_path"]
)
test_df["label"] = test_df["image_path"].apply(create_label)
test_df.to_csv("data/flowers/test.csv", index=False)

# [STEP 4] Clean the cloned folder
shutil.rmtree("Unlocked_Challenge_4")
