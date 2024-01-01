import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
df = pd.read_csv("data/dr-data-kaggle/trainLabels.csv.zip")

# add prefix to the image path: "data/dr-data-kaggle/images"
df["image"] = "data/dr-data-kaggle/images/" + df["image"] + ".jpeg"

# rename the column "level" to "label"
df.rename(columns={"image": "image_path", "level": "label"}, inplace=True)

# split the data into train and test
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"], shuffle=True
)


# save the data
train_df.to_csv("data/dr-data-kaggle/train.csv", index=False)
test_df.to_csv("data/dr-data-kaggle/test.csv", index=False)
