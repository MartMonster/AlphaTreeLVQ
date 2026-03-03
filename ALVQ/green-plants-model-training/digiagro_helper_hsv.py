from os import listdir
import cv2
import numpy as np


hue = [30, 50]
saturation = [110, 240]

def read_imgs(img_dir, class_subdirs):
    n_classes = len(class_subdirs)
    images = []
    
    for i in range(n_classes):
        class_images = []
        for subdir, weight in class_subdirs[i]:
            path = img_dir + '/' + subdir + '/'
            files = [path + f for f in listdir(path) if f.endswith('.png')]
            if len(files) == 0:
                continue
            weight_img = weight / len(files)
            for f in files:
                img = cv2.imread(f)
                if img is None:
                    continue
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # Apply mask
                mask = (
                    (hsv[:, :, 0] >= hue[0]) & (hsv[:, :, 0] <= hue[1]) &
                    (hsv[:, :, 1] >= saturation[0]) & (hsv[:, :, 1] <= saturation[1])
                )
                # Extract only H and S values where mask is True
                hs_pixels = hsv[:, :, 0:2].reshape(-1, 2)
                # Skip image if no valid pixels
                if hs_pixels.shape[0] == 0:
                    continue
                class_images.append((hs_pixels, weight_img))
        images.append(class_images)
    images = [cls for cls in images if len(cls) > 0]
    return images


def randomize_sample_remainders(n_samples_array, n_samples_total):
    result = (n_samples_array).astype(np.int64)
    diff = np.int64(n_samples_total) - result.sum()

    if diff == 0:
        return result

    while diff > 0:
        idx_todo = result < n_samples_array
        arr = n_samples_array[idx_todo] - result[idx_todo]
        arr = arr / arr.sum() * diff
        arr = np.concatenate(([0], arr))
        arr = np.cumsum(arr)
        rnd = np.random.random_sample(diff) + np.arange(diff)
        n_items = arr.shape[0] - 1
        rnd_idx0 = np.minimum(arr[0:n_items].astype(np.int64), diff - 1)
        rnd_idx1 = np.minimum(arr[1:(n_items+1)].astype(np.int64), diff - 1)

        to_add = ((rnd[rnd_idx0] >= arr[0:n_items]) &
                  (rnd[rnd_idx0] < arr[1:(n_items+1)]))

        to_add |= ((rnd[rnd_idx1] >= arr[0:n_items]) &
                   (rnd[rnd_idx1] < arr[1:(n_items+1)]))

        result[idx_todo] += to_add
        diff = np.int64(n_samples_total) - result.sum()

    return result

def img_sample_counts(images, n_samples):
    n_classes = len(images)
    weight_sums = np.zeros(n_classes)

    for i in range(n_classes):
        for img, weight in images[i]:
            weight_sums[i] += weight

    if weight_sums.sum() == 0:
        raise RuntimeError("No valid masked pixels found in dataset.")
    weight_sums = weight_sums / weight_sums.sum()
    sample_count_classes = n_samples * weight_sums
    sample_count_classes = randomize_sample_remainders(
        sample_count_classes, n_samples
    )

    sample_count = []
    for i in range(n_classes):
        n_images = len(images[i])
        sample_count_images = np.zeros(n_images)

        for k, (img, weight) in enumerate(images[i]):
            sample_count_images[k] = weight
        if sample_count_images.sum() == 0:
            sample_count.append(sample_count_images)
            continue
        sample_count_images = (
            sample_count_images /
            sample_count_images.sum() *
            sample_count_classes[i]
        )
        sample_count_images = randomize_sample_remainders(
            sample_count_images,
            sample_count_classes[i]
        )
        sample_count.append(sample_count_images)

    return sample_count

def xval(imgs, val_fraction = 0.2):
    imgs_train = []
    imgs_val = []
    n_classes = len(imgs)
    
    for i in range(n_classes):    
        n_images = len(imgs[i])
        if n_images == 0:
            imgs_train.append([])
            imgs_val.append([])
            continue
        indices = np.random.permutation(np.arange(n_images))
        n_val = max(np.int64(n_images * val_fraction), 1)
        val = []
        for k in range(n_val):
            val.append(imgs[i][indices[k]])

        train = []
        for k in range(n_images - n_val):
            train.append(imgs[i][indices[n_val + k]])
                
        imgs_train.append(train)
        imgs_val.append(val)
    
    return imgs_train, imgs_val

def img_sample(hs_pixels, n_samples):
    n_available = hs_pixels.shape[0]
    if n_available == 0:
        return np.empty((0, 2), dtype=np.uint8)
    idx = np.random.randint(0, n_available, n_samples)
    return hs_pixels[idx]

def create_samples(images, batch_sz):
    sample_counts = img_sample_counts(images, batch_sz)

    pix = np.empty((batch_sz, 2), dtype=np.uint8)
    labels = np.empty(batch_sz, dtype=np.uint8)
    n_classes = len(images)
    offset = 0

    for i in range(n_classes):
        for k, (hs_pixels, weight) in enumerate(images[i]):
            count = sample_counts[i][k]
            if count == 0:
                continue
            img_pix = img_sample(hs_pixels, count)
            pix[offset:(offset + count), :] = img_pix
            labels[offset:(offset + count)] = i
            offset += count

    if offset != batch_sz:
        raise RuntimeError("Batch size mismatch after sampling.")
    I = np.random.permutation(batch_sz)

    return pix[I, ...], labels[I]

def data_loader(images, batch_sz):
    while True:
        samples, labels = create_samples(images, batch_sz)

        samples = samples.astype(np.float32)
        # print(samples)
        # OpenCV HSV ranges:
        # H: 0-179
        # S: 0-255
        samples[:, 0] /= 179.0
        samples[:, 1] /= 255.0
        if np.isnan(samples).any():
            print("NaN detected in samples")
            print("samples:", samples)
            raise RuntimeError("NaN in samples")
        yield samples, labels


def feats_mean_std(loader, n_batches):
    stats_mu = []
    stats_M2 = []

    for batch_nr in range(n_batches):
        X, y = next(loader)
        batch_sz = X.shape[0]

        M2 = np.var(X, axis=0) * batch_sz
        mu = np.mean(X, axis=0)

        stats_mu.append(mu)
        stats_M2.append(M2)

    mu = np.zeros(stats_mu[0].shape[0], dtype=np.float64)
    M2 = np.zeros(stats_M2[0].shape[0], dtype=np.float64)

    for i in range(len(stats_mu)):
        delta = stats_mu[i] - mu
        mu += delta / (i + 1.0)
        M2 += stats_M2[i] + delta * delta * i / (i + 1.0)

    return (
        mu.astype(np.float32),
        np.sqrt(M2 / (batch_sz * n_batches - 1)).astype(np.float32)
    )