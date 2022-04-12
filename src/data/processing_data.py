import os

def data_processor(dti):
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(f"{dti}/PetImages", folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print("Deleted %d images" % num_skipped)

    image_size = (180, 180)
    batch_size = 32
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                                                                  f"{dti}/PetImages",
                                                                  validation_split=0.2,
                                                                  subset="training",
                                                                  seed=1337, 
                                                                  image_size=image_size,
                                                                  batch_size=batch_size,
                                                                  )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                                                                f"{dti}/PetImages",
                                                                validation_split=0.2,
                                                                subset="validation",
                                                                seed=1337,
                                                                image_size=image_size,
                                                                batch_size=batch_size,
                                                                )
    
