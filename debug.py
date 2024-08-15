import glob
import os
import torch.utils.data as data

from training.dataset import Dataset

DATASET_DIR = "./maestro-v3.0.0-preprocessed"

def main():
    dataset = Dataset(
        features_dir=os.path.join(DATASET_DIR, "features"),
        labels_dir=os.path.join(DATASET_DIR, "labels"),
        filenames=[
            os.path.basename(f)
            for f in glob.glob(os.path.join(DATASET_DIR, "features", "*.pt"))
        ],
    )

    dataloader = data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.collate_fn,
    )

    for batch in dataloader:
        print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape, batch[4].shape)
        break

if __name__ == "__main__":
    main()