import time
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

time_start = time.time()
# Configuration
MAX_EPOCHS = 100
SUBJECTS = [1, 2, 3]
SEEDS = [0, 1, 2]
all_types = ["torch", "old_fixed", "old_nonfixed"]
low_cut_hz = 4.0  # low cut frequency for filtering
high_cut_hz = 38.0  # high cut frequency for filtering
factor_new = 1e-3
init_block_size = 1000
factor = 1e6
n_classes = 4

# Preprocessing pipeline
preprocessors = [
    Preprocessor("pick_types", eeg=True, meg=False, stim=False),
    Preprocessor(lambda data: multiply(data, factor)),
    Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),
    Preprocessor(
        exponential_moving_standardize,
        factor_new=factor_new,
        init_block_size=init_block_size,
    ),
]

cuda = torch.cuda.is_available()
mps = torch.backends.mps.is_available()
device = "cuda" if cuda else "mps" if mps else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True

# 1. Pre-load and Preprocess Data for all subjects
print("Pre-loading and preprocessing data...")
data_by_subject = {}
n_chans = None
input_window_samples = None

for subject_id in SUBJECTS:
    print(f"Processing Subject {subject_id}...")
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])
    preprocess(dataset, preprocessors)

    trial_start_offset_seconds = -0.5
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
    )

    splitted = windows_dataset.split("session")
    train_set = splitted["0train"]
    valid_set = splitted["1test"]

    data_by_subject[subject_id] = (train_set, valid_set)

    # Store dimensions from the first subject
    if n_chans is None:
        n_chans = train_set[0][0].shape[0]
        input_window_samples = windows_dataset[0][0].shape[1]

print("Data preparation complete.")


results = {}

# 2. Main Experiment Loop
for attention_type in all_types:
    print(f"Running experiments for attention_type={attention_type}")

    type_dfs = []

    for subject_id in SUBJECTS:
        train_set, valid_set = data_by_subject[subject_id]

        for seed in SEEDS:
            print(f"  Subject={subject_id}, Seed={seed}")
            set_random_seeds(seed=seed, cuda=cuda)

            model = EEGConformer(
                n_chans=n_chans,
                n_outputs=n_classes,
                final_fc_length=2760,
                attention_type=attention_type,
            )

            if cuda:
                model.cuda()
            elif mps:
                model.to("mps")

            # Hyperparameters
            lr = 0.0625 * 0.01
            weight_decay = 0
            batch_size = 64
            n_epochs = MAX_EPOCHS

            clf = EEGClassifier(
                model,
                criterion=torch.nn.CrossEntropyLoss,
                optimizer=torch.optim.AdamW,
                train_split=predefined_split(valid_set),
                optimizer__lr=lr,
                optimizer__weight_decay=weight_decay,
                batch_size=batch_size,
                max_epochs=n_epochs,
                classes=[0, 1, 2, 3],
                callbacks=[
                    "accuracy",
                    (
                        "lr_scheduler",
                        LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1),
                    ),
                ],
                device=device,
                verbose=0,
            )

            clf.fit(train_set, y=None)

            # Extract history
            results_columns = [
                "train_loss",
                "valid_loss",
                "train_accuracy",
                "valid_accuracy",
            ]
            df = pd.DataFrame(
                clf.history[:, results_columns],
                columns=results_columns,
                index=clf.history[:, "epoch"],
            )

            df = df.assign(
                train_misclass=100 - 100 * df.train_accuracy,
                valid_misclass=100 - 100 * df.valid_accuracy,
            )
            type_dfs.append(df)

    # Aggregate results for this attention_type
    concat_df = pd.concat(type_dfs)
    by_row_index = concat_df.groupby(concat_df.index)
    df_mean = by_row_index.mean()
    df_std = by_row_index.std()

    results[attention_type] = {"mean": df_mean, "std": df_std}

output_dir = Path("temp")
output_dir.mkdir(parents=True, exist_ok=True)

title_suffix = f"({len(SUBJECTS)} Subjects, {len(SEEDS)} Seeds per subject)"

# Plot Loss
fig_loss, ax_loss = plt.subplots(figsize=(10, 6))

# distinct colors for Train vs Valid
colors_dict = {
    "torch": ("tab:blue", "cornflowerblue"),
    "old_fixed": ("tab:orange", "sandybrown"),
    "old_nonfixed": ("tab:green", "limegreen"),
}

offset_step = 0.05  # Slight shift to avoid overlap

for i, attention_type in enumerate(all_types):
    res = results[attention_type]
    df_mean = res["mean"]
    df_std = res["std"]
    c_train, c_valid = colors_dict[attention_type]

    epochs = df_mean.index
    x_offset = (i - 1) * offset_step

    # Train Loss
    ax_loss.errorbar(
        epochs + x_offset,
        df_mean["train_loss"],
        yerr=df_std["train_loss"],
        fmt="-o",
        color=c_train,
        label=f"{attention_type} Train",
        capsize=3,
        alpha=0.9,
    )

    # Valid Loss
    ax_loss.errorbar(
        epochs + x_offset,
        df_mean["valid_loss"],
        yerr=df_std["valid_loss"],
        fmt=":x",
        color=c_valid,
        label=f"{attention_type} Valid",
        capsize=3,
        alpha=0.9,
    )

ax_loss.set_xlabel("Epoch", fontsize=14)
ax_loss.set_ylabel("Loss", fontsize=14)
ax_loss.legend(fontsize=12)
ax_loss.set_title(
    f"Training and Validation Loss Comparison\n{title_suffix}", fontsize=16
)
ax_loss.grid(True)
fig_loss.tight_layout()
fig_loss.savefig(output_dir / "loss_comparison.png", dpi=300)
plt.close(fig_loss)

# Plot Misclassification
fig_misc, ax_misc = plt.subplots(figsize=(10, 6))

for i, attention_type in enumerate(all_types):
    res = results[attention_type]
    df_mean = res["mean"]
    df_std = res["std"]
    c_train, c_valid = colors_dict[attention_type]

    epochs = df_mean.index
    x_offset = (i - 1) * offset_step

    # Train Misclassification
    ax_misc.errorbar(
        epochs + x_offset,
        df_mean["train_misclass"],
        yerr=df_std["train_misclass"],
        fmt="-o",
        color=c_train,
        label=f"{attention_type} Train",
        capsize=3,
        alpha=0.9,
    )

    # Valid Misclassification
    ax_misc.errorbar(
        epochs + x_offset,
        df_mean["valid_misclass"],
        yerr=df_std["valid_misclass"],
        fmt=":x",
        color=c_valid,
        label=f"{attention_type} Valid",
        capsize=3,
        alpha=0.9,
    )

ax_misc.set_xlabel("Epoch", fontsize=14)
ax_misc.set_ylabel("Misclassification Rate [%]", fontsize=14)
ax_misc.legend(fontsize=12)
ax_misc.set_title(f"Misclassification Rate Comparison\n{title_suffix}", fontsize=16)
ax_misc.grid(True)
fig_misc.tight_layout()
fig_misc.savefig(output_dir / "misclass_comparison.png", dpi=300)
plt.close(fig_misc)
print(f"Experiment completed in (minutes) {(time.time() - time_start) / 60:.2f}")
