from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from numpy import multiply
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.datasets import MOABBDataset
from braindecode.models import EEGConformer
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from braindecode.util import set_random_seeds

subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])


low_cut_hz = 4.0  # low cut frequency for filtering
high_cut_hz = 38.0  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000
# Factor to convert from V to uV
factor = 1e6

preprocessors = [
    Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
    Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(
        exponential_moving_standardize,  # Exponential moving standardization
        factor_new=factor_new,
        init_block_size=init_block_size,
    ),
]

# Transform the data
preprocess(dataset, preprocessors)


trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info["sfreq"]
assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)


splitted = windows_dataset.split("session")
train_set = splitted["0train"]
valid_set = splitted["1test"]


cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
mps = torch.backends.mps.is_available()
device = "cuda" if cuda else "mps" if mps else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True

seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
# Extract number of chans and time steps from dataset
n_chans = train_set[0][0].shape[0]
input_window_samples = windows_dataset[0][0].shape[1]

all_types = ["torch", "old_fixed", "old_nonfixed"]

results = {}

for attention_type in all_types:
    print(f"Training model with attention_type={attention_type}")
    model = EEGConformer(
        n_chans=n_chans,
        n_outputs=n_classes,
        final_fc_length=2760,
        attention_type=attention_type,
    )

    # Send model to GPU
    if cuda:
        model.cuda()
    elif mps:
        model.to("mps")

    # These values we found good for shallow network:
    lr = 0.0625 * 0.01
    weight_decay = 0

    # For deep4 they should be:
    # lr = 1 * 0.01
    # weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 100

    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),  # using valid_set for validation
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=n_epochs,
        classes=[0, 1, 2, 3],
        callbacks=[
            "accuracy",
            ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
        ],
        device=device,
    )
    # Model training for a specified number of epochs. `y` is None as it is already supplied
    # in the dataset.
    clf.fit(train_set, y=None)

    # Extract loss and accuracy values for plotting from history object
    results_columns = ["train_loss", "valid_loss", "train_accuracy", "valid_accuracy"]
    df = pd.DataFrame(
        clf.history[:, results_columns],
        columns=results_columns,  # type:ignore
        index=clf.history[:, "epoch"],
    )

    # get percent of misclass for better visual comparison to loss
    df = df.assign(
        train_misclass=100 - 100 * df.train_accuracy,
        valid_misclass=100 - 100 * df.valid_accuracy,
    )
    results[attention_type] = df

output_dir = Path("temp")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot Loss
fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
colors = ["tab:blue", "tab:orange", "tab:green"]

for i, attention_type in enumerate(all_types):
    df = results[attention_type]
    color = colors[i % len(colors)]

    ax_loss.plot(
        df.index,
        df["train_loss"],
        linestyle="-",
        marker="o",
        color=color,
        label=f"{attention_type} Train",
    )
    ax_loss.plot(
        df.index,
        df["valid_loss"],
        linestyle=":",
        marker="x",
        color=color,
        label=f"{attention_type} Valid",
    )

ax_loss.set_xlabel("Epoch", fontsize=14)
ax_loss.set_ylabel("Loss", fontsize=14)
ax_loss.legend(fontsize=12)
ax_loss.set_title("Training and Validation Loss Comparison", fontsize=16)
ax_loss.grid(True)
fig_loss.tight_layout()
fig_loss.savefig(output_dir / "loss_comparison.png", dpi=300)
plt.close(fig_loss)

# Plot Misclassification
fig_misc, ax_misc = plt.subplots(figsize=(10, 6))

for i, attention_type in enumerate(all_types):
    df = results[attention_type]
    color = colors[i % len(colors)]

    ax_misc.plot(
        df.index,
        df["train_misclass"],
        linestyle="-",
        marker="o",
        color=color,
        label=f"{attention_type} Train",
    )
    ax_misc.plot(
        df.index,
        df["valid_misclass"],
        linestyle=":",
        marker="x",
        color=color,
        label=f"{attention_type} Valid",
    )

ax_misc.set_xlabel("Epoch", fontsize=14)
ax_misc.set_ylabel("Misclassification Rate [%]", fontsize=14)
ax_misc.legend(fontsize=12)
ax_misc.set_title("Misclassification Rate Comparison", fontsize=16)
ax_misc.grid(True)
fig_misc.tight_layout()
fig_misc.savefig(output_dir / "misclass_comparison.png", dpi=300)
plt.close(fig_misc)
