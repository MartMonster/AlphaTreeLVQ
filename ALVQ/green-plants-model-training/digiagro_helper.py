from os import listdir
from os.path import isfile, join
import cv2
import numpy as np

def read_imgs(img_dir, class_subdirs):
    n_classes = len(class_subdirs)
    images = []
    
    for i in range(n_classes):
        class_images = []
        for subdir, weight in class_subdirs[i]:
            path = img_dir + '/' + subdir + '/'
            files = [path + f for f in listdir(path) if f.endswith('.png')]
            weight_img = weight / len(files)
            for f in files:
                img = cv2.imread(f)                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).squeeze()
                class_images.append((img, weight_img))
        images.append(class_images)

    return images


def randomize_sample_remainders(n_samples_array, n_samples_total):
    n_items = n_samples_array.shape[0]
    result = (n_samples_array).astype(np.int64)

    diff = np.int64(n_samples_total) - result.sum()    

    if diff == 0:
        return result

    # Should take only few iterations
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
        # Check if random number is in the range
        to_add = ((rnd[rnd_idx0] >= arr[0:n_items]) & (rnd[rnd_idx0] < arr[1:(n_items+1)]))
        to_add |= ((rnd[rnd_idx1] >= arr[0:n_items]) & (rnd[rnd_idx1] < arr[1:(n_items+1)]))
        result[idx_todo] += to_add
        diff = np.int64(n_samples_total) - result.sum()    

    assert(diff == 0)

    return result

def img_sample_counts(images, n_samples):
    n_classes = len(images)
    weight_sums = np.zeros(n_classes)
    
    for i in range(n_classes):
        for img, weight in images[i]:
            weight_sums[i] += weight

    weight_sums = weight_sums / weight_sums.sum()
    sample_count_classes = n_samples * weight_sums
    sample_count_classes = randomize_sample_remainders(sample_count_classes, n_samples)

    sample_count = []
    for i in range(n_classes):
        n_images = len(images[i])
        sample_count_images = np.zeros(n_images)
        
        for k, (img, weight) in enumerate(images[i]):
            sample_count_images[k] = weight

        sample_count_images = sample_count_images / sample_count_images.sum() * sample_count_classes[i]
        sample_count_images = randomize_sample_remainders(sample_count_images, sample_count_classes[i])        
        sample_count.append(sample_count_images)

    return sample_count
        
def xval(imgs, val_fraction = 0.2):
    imgs_train = []
    imgs_val = []
    n_classes = len(imgs)
    
    for i in range(n_classes):    
        n_images = len(imgs[i])
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

def img_sample(img, n_samples):
    height = img.shape[0]
    width = img.shape[1]
    
    Y = np.random.randint(0, height, n_samples)
    X = np.random.randint(0, width, n_samples)

    return img[Y, X, 0:3].reshape(-1, 3)
    
def create_samples(images, batch_sz):
    sample_counts = img_sample_counts(images, batch_sz)
    
    pix = np.empty((batch_sz, 3), dtype=np.uint8)
    labels = np.empty(batch_sz, dtype=np.uint8)
    n_classes = len(images)
    offset = 0
    
    for i in range(n_classes):
        n_images = len(images[i])
        
        for k, (img, weight) in enumerate(images[i]):
            count = sample_counts[i][k]
            img_pix = img_sample(img, count)
            pix[offset:(offset + count), :] = img_pix
            labels[offset:(offset + count)] = i
            offset += count

    assert(offset == batch_sz)
        
    I = np.random.permutation(batch_sz)

    return pix[I, ...], labels[I]
        
def data_loader(images, batch_sz):
    while True:    
        samples, labels = create_samples(images, batch_sz)
        samples = samples.reshape(-1, 3).astype(np.float32) * (1/255)
        yield samples, labels 


def feats_mean_std(loader, n_batches):
    stats_mu = []
    stats_M2 = []

    for batch_nr in range(n_batches):
        X, y = next(loader)
        batch_sz = X.shape[0]

        M2 = np.var(X, axis = 0) * batch_sz
        mu = np.mean(X, axis = 0)

        stats_mu.append(mu)
        stats_M2.append(M2)

    mu = np.zeros(stats_mu[0].shape[0], dtype=np.float64)
    M2 = np.zeros(stats_M2[0].shape[0], dtype=np.float64)
    
    for i in range(len(stats_mu)):
        delta = stats_mu[i] - mu        
        mu += delta / (i + 1.0)
        M2 += stats_M2[i] + delta * delta * i / (i + 1.0)
            
    return mu.astype(np.float32), np.sqrt(M2 / (batch_sz * n_batches - 1)).astype(np.float32)