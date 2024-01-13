import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="Prepare the DR dataset.")
parser.add_argument(
    "--num-samples",
    type=int,
    default=100,
    help="The number of samples per class.",
)
parser.add_argument(
    "--test-size",
    type=float,
    default=0.2,
    help="The test size.",
)
args = parser.parse_args()

# Read the data
df = pd.read_csv("data/dr-data-kaggle/trainLabels.csv.zip")

# add prefix to the image path: "data/dr-data-kaggle/images"
df["image"] = "data/dr-data-kaggle/images/" + df["image"] + ".jpeg"

# rename the column "level" to "label"
df.rename(columns={"image": "image_path", "level": "label"}, inplace=True)

# drop extra data from the dataframe by limiting max of 100 samples from each class
df = df.groupby("label").head(args.num_samples).reset_index(drop=True)

# plot the distribution of the labels
df["label"].value_counts().plot.bar()
# save the plot
plt.savefig("data/dr-data-kaggle/label_distribution.png")

# split the data into train and test
train_df, test_df = train_test_split(
    df, test_size=args.test_size, random_state=42, stratify=df["label"], shuffle=True
)

# print data distribution
print("Train data distribution:")
print(train_df["label"].value_counts())
print("Test data distribution:")
print(test_df["label"].value_counts())

# save the data
train_df.to_csv("data/dr-data-kaggle/train.csv", index=False)
test_df.to_csv("data/dr-data-kaggle/test.csv", index=False)
