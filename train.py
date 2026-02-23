import os
import json
import random
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.applications import DenseNet201


# -----------------------------
# Fixed class order (Raabin-WBC)
# -----------------------------
CLASS_NAMES = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
NUM_CLASSES = len(CLASS_NAMES)
class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

# Binary grouping:
AGRAN = {"Lymphocyte", "Monocyte"}                 # 0
GRAN  = {"Neutrophil", "Eosinophil", "Basophil"}   # 1
agran_idx = set(class_to_idx[c] for c in AGRAN)
gran_idx  = set(class_to_idx[c] for c in GRAN)

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def gpu_setup():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[INFO] GPU detected: {gpus[0].name}")
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except Exception as e:
            print("[WARN] Could not set memory growth:", e)
    else:
        print("[WARN] No GPU found. Training will run on CPU.")


def build_file_list(data_dir: str):
    image_paths, labels = [], []
    for c in CLASS_NAMES:
        cdir = os.path.join(data_dir, c)
        if not os.path.isdir(cdir):
            raise FileNotFoundError(f"Missing class folder: {cdir}")

        files = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.lower().endswith(VALID_EXT)]
        image_paths.extend(files)
        labels.extend([class_to_idx[c]] * len(files))
        print(f"[INFO] {c}: {len(files)} images")

    return np.array(image_paths), np.array(labels, dtype=np.int32)


def to_binary_tf(y):
    gran_list = tf.constant(sorted(list(gran_idx)), dtype=y.dtype)
    is_gran = tf.reduce_any(tf.equal(tf.expand_dims(y, -1), gran_list), axis=-1)
    return tf.cast(is_gran, tf.int32)


def make_dataset(paths, labels, img_size, batch_size, seed, training=True):
    def preprocess(path, y5):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0

        y5 = tf.cast(y5, tf.int32)
        y2 = to_binary_tf(y5)
        return img, {"wbc_5class": y5, "agran_gran": y2}

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def SAM(x, kernel_size=7):
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])  # (H,W,2)
    attn = layers.Conv2D(1, kernel_size=kernel_size, padding="same", activation="sigmoid")(concat)
    return layers.Multiply()([x, attn])


def CAM(x, reduction=16):
    channels = x.shape[-1]
    if channels is None:
        raise ValueError("Channel dimension is None; cannot build CAM.")
    hidden = max(channels // reduction, 1)

    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)

    shared_dense1 = layers.Dense(hidden, activation="relu")
    shared_dense2 = layers.Dense(channels, activation="sigmoid")

    gap_out = shared_dense2(shared_dense1(gap))
    gmp_out = shared_dense2(shared_dense1(gmp))

    attn = layers.Add()([gap_out, gmp_out])
    attn = layers.Reshape((1, 1, channels))(attn)
    return layers.Multiply()([x, attn])


def build_model(input_shape, train_backbone=True):
    inputs = layers.Input(shape=input_shape)
    backbone = DenseNet201(include_top=False, weights="imagenet", input_tensor=inputs)
    backbone.trainable = train_backbone

    x = backbone.output  # (7,7,1920)
    x = layers.Add()([SAM(x), CAM(x)])  # mixed attention

    x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    out5 = layers.Dense(NUM_CLASSES, activation="softmax", name="wbc_5class")(x)
    out2 = layers.Dense(2, activation="softmax", name="agran_gran")(x)

    return models.Model(inputs, [out5, out2], name="DenseNet201_MixedAttention_MultiTask")


def save_curves(history, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    epochs = list(range(1, len(history.history["loss"]) + 1))

    def plot_and_save(y_train, y_val, title, ylabel, fname):
        plt.figure()
        plt.plot(epochs, y_train, label="Train")
        plt.plot(epochs, y_val, label="Val")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(fig_dir, fname)
        plt.savefig(path, dpi=300)
        plt.show()
        print("[INFO] Saved:", path)

    if "wbc_5class_accuracy" in history.history:
        plot_and_save(history.history["wbc_5class_accuracy"],
                      history.history["val_wbc_5class_accuracy"],
                      "5-class Accuracy", "Accuracy", "acc_5class.png")

    if "agran_gran_accuracy" in history.history:
        plot_and_save(history.history["agran_gran_accuracy"],
                      history.history["val_agran_gran_accuracy"],
                      "2-class Accuracy", "Accuracy", "acc_2class.png")

    plot_and_save(history.history["loss"],
                  history.history["val_loss"],
                  "Total Loss", "Loss", "loss_total.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Raabin-WBC root folder")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224], help="Image size H W")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_size", type=int, default=13305)
    parser.add_argument("--test_size", type=int, default=1664)
    parser.add_argument("--train_backbone", action="store_true", help="Fine-tune DenseNet backbone (default: False)")
    args = parser.parse_args()

    set_seed(args.seed)
    gpu_setup()

    out_dir = args.out_dir
    split_dir = os.path.join(out_dir, "splits")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    img_size = tuple(args.img_size)

    # File list
    all_images, all_labels = build_file_list(args.data_dir)
    print("[INFO] Total images:", len(all_images))

    if len(all_images) < (args.train_size + args.test_size + 1):
        raise ValueError("Dataset too small for fixed split sizes. Adjust --train_size/--test_size.")

    # Split
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        all_images, all_labels,
        train_size=args.train_size,
        random_state=args.seed,
        stratify=all_labels
    )

    val_size = len(temp_imgs) - args.test_size
    if val_size <= 0:
        raise ValueError("Validation size <= 0. Reduce --test_size or --train_size.")

    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=temp_labels
    )

    print(f"[INFO] Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    # Save split artifacts
    split_info = {
        "dataset": "Raabin-WBC",
        "seed": args.seed,
        "img_size": list(img_size),
        "batch_size": args.batch_size,
        "train_size": int(len(train_imgs)),
        "val_size": int(len(val_imgs)),
        "test_size": int(len(test_imgs)),
        "class_names": CLASS_NAMES,
        "binary_mapping": {
            "Agranulocytes(0)": sorted(list(AGRAN)),
            "Granulocytes(1)": sorted(list(GRAN)),
        }
    }
    with open(os.path.join(split_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    np.save(os.path.join(split_dir, "train_paths.npy"), train_imgs)
    np.save(os.path.join(split_dir, "val_paths.npy"), val_imgs)
    np.save(os.path.join(split_dir, "test_paths.npy"), test_imgs)
    np.save(os.path.join(split_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(split_dir, "val_labels.npy"), val_labels)
    np.save(os.path.join(split_dir, "test_labels.npy"), test_labels)

    # Datasets
    train_ds = make_dataset(train_imgs, train_labels, img_size, args.batch_size, args.seed, training=True)
    val_ds   = make_dataset(val_imgs, val_labels, img_size, args.batch_size, args.seed, training=False)
    test_ds  = make_dataset(test_imgs, test_labels, img_size, args.batch_size, args.seed, training=False)

    # Model
    model = build_model(input_shape=(img_size[0], img_size[1], 3), train_backbone=args.train_backbone)
    model.summary()

    optimizer = AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    model.compile(
        optimizer=optimizer,
        loss={"wbc_5class": "sparse_categorical_crossentropy",
              "agran_gran": "sparse_categorical_crossentropy"},
        loss_weights={"wbc_5class": 1.0, "agran_gran": 0.3},
        metrics={"wbc_5class": ["accuracy"], "agran_gran": ["accuracy"]}
    )

    best_model_path = os.path.join(out_dir, "best_model.keras")
    csv_log_path = os.path.join(out_dir, "training_log.csv")

    callbacks = [
        ModelCheckpoint(best_model_path, monitor="val_wbc_5class_accuracy",
                        save_best_only=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        CSVLogger(csv_log_path)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Final quick test evaluation
    results = model.evaluate(test_ds, verbose=1)
    for n, v in zip(model.metrics_names, results):
        print(f"[RESULT] {n}: {v:.6f}")

    # Save training curves
    save_curves(history, fig_dir)
    print("[DONE] Outputs saved in:", out_dir)


if __name__ == "__main__":
    main()
