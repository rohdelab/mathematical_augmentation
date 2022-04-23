import numpy as np
from scipy.io import loadmat
import os
import h5py
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage import transform
from einops import rearrange


def take_affine_tform_aug(xAffT, yAffT,img_size,ards, N_aug):
    # xAffT: data to be aug --> shape ()
    # yAffT: label to be aug --> shape ()
    # N_aug: number of aug --> int
    
    if ards=='SYNTH':
        sx=np.random.uniform(0.5,2.0)
        sy=np.random.uniform(0.5,2.0)
        tx=np.random.uniform(-15,15)
        ty=np.random.uniform(-15,15)
        sh_ang=np.random.uniform(-np.pi/5,np.pi/5)
        rot_ang=np.random.uniform(-40,40)
    elif ards=='AFFNISTb_out':
        sx=np.random.uniform(0.8,1.2)
        sy=np.random.uniform(0.8,1.2)
        tx=np.random.uniform(-5,5)
        ty=np.random.uniform(-5,5)
        sh_ang=np.random.uniform(-np.pi/20,np.pi/20)
        rot_ang=np.random.uniform(-10,10)
    else:
        sx=np.random.uniform(0.5,2.0)
        sy=np.random.uniform(0.5,2.0)
        tx=np.random.uniform(-15,15)
        ty=np.random.uniform(-15,15)
        sh_ang=np.random.uniform(-np.pi/10,np.pi/10)
        rot_ang=np.random.uniform(-20,20)

    xAffTAug = []
    yAffTAug = []
    for eachAug in range(N_aug):
        for eachX, eachY in zip(xAffT, yAffT):

            eachXPadTform=eachX
            
            szs=2*int(np.round(img_size/2))

            eachXPadTform = np.pad(eachXPadTform, ((0,szs),(0,szs), (0,0)), 'constant',constant_values=-1.0)
            
            tform_s = transform.AffineTransform(scale=(sx, sy), shear=sh_ang)
            eachXPadTform = transform.warp(eachXPadTform, tform_s.inverse)  
            
            cent=szs
            cy1=sx*cent/2; cx1=sy*cent/2;
            
            tform_t1 = transform.AffineTransform(translation=(cent-cy1, cent-cx1)) 
            eachXPadTform = transform.warp(eachXPadTform, tform_t1.inverse) 
            
            eachXPadTform = transform.rotate(eachXPadTform,rot_ang)
            
            eachXPadTform = np.pad(eachXPadTform, ((15,15),(15,15), (0,0)), 'constant',constant_values=-1.0)
            tform_t = transform.AffineTransform(translation=(tx, ty)) 
            eachXPadTform = transform.warp(eachXPadTform, tform_t.inverse)  
            
            # tform = transform.AffineTransform(scale=(sx, sy), rotation=rot_ang, translation=(tx, ty), shear=sh_ang)
            # eachXPad = np.pad(eachX, ((18,18),(18,18), (0,0)), 'constant') #((top, bottom), (left, right)) https://stackoverflow.com/questions/38191855/zero-pad-numpy-array
            # eachXPadTform = transform.warp(eachXPad, tform.inverse) 
            
            
            xAffTAug.append(eachXPadTform)
            yAffTAug.append(eachY)

    xAffTAug = np.asarray(xAffTAug)
    yAffTAug = np.asarray(yAffTAug) 

    return xAffTAug, yAffTAug

def take_affine_tform_augV2(xAffT, yAffT,img_size,ards,N_aug):
    # xAffT: data to be aug --> shape ()
    # yAffT: label to be aug --> shape ()
    # N_aug: number of aug --> int
    
        
    xAffTAug = []
    yAffTAug = []
    for eachAug in range(N_aug):
        for eachX, eachY in zip(xAffT, yAffT):
            
            
            if ards=='SYNTH':
                sx=np.random.uniform(0.5,2.0)
                sy=np.random.uniform(0.5,2.0)
                tx=np.random.uniform(-15,15)
                ty=np.random.uniform(-15,15)
                sh_ang=np.random.uniform(-np.pi/5,np.pi/5)
                rot_ang=np.random.uniform(-40,40)
            elif ards=='AFFNISTb_out':
                sx=np.random.uniform(0.8,1.2)
                sy=np.random.uniform(0.8,1.2)
                tx=np.random.uniform(-5,5)
                ty=np.random.uniform(-5,5)
                sh_ang=np.random.uniform(-np.pi/20,np.pi/20)
                rot_ang=np.random.uniform(-10,10)
            else:
                sx=np.random.uniform(0.5,2.0)
                sy=np.random.uniform(0.5,2.0)
                tx=np.random.uniform(-15,15)
                ty=np.random.uniform(-15,15)
                sh_ang=np.random.uniform(-np.pi/10,np.pi/10)
                rot_ang=np.random.uniform(-20,20)
            
            
            eachXPadTform=eachX+1
            
            szs=2*int(np.round(img_size/2))

            #eachXPadTform = np.pad(eachXPadTform, ((0,szs),(0,szs), (0,0)), 'constant',constant_values=-1)
            eachXPadTform = np.pad(eachXPadTform, ((0,szs),(0,szs), (0,0)), 'constant')
            
            tform_s = transform.AffineTransform(scale=(sx, sy), shear=sh_ang)
            eachXPadTform = transform.warp(eachXPadTform, tform_s.inverse)  
            
            cent=szs
            cy1=sx*cent/2; cx1=sy*cent/2;
            
            tform_t1 = transform.AffineTransform(translation=(cent-cy1, cent-cx1)) 
            eachXPadTform = transform.warp(eachXPadTform, tform_t1.inverse) 
            
            eachXPadTform = transform.rotate(eachXPadTform,rot_ang)
            
            #eachXPadTform = np.pad(eachXPadTform, ((15,15),(15,15), (0,0)), 'constant',constant_values=-1)
            eachXPadTform = np.pad(eachXPadTform, ((15,15),(15,15), (0,0)), 'constant')
            
            tform_t = transform.AffineTransform(translation=(tx, ty)) 
            eachXPadTform = transform.warp(eachXPadTform, tform_t.inverse)  
            eachXPadTform=eachXPadTform-1

            xAffTAug.append(eachXPadTform)
            yAffTAug.append(eachY)

    xAffTAug = np.asarray(xAffTAug)
    yAffTAug = np.asarray(yAffTAug) 

    return xAffTAug, yAffTAug

def take_affine_tform_augV2trad(xAffT, yAffT,img_size,ards,N_aug):
    # xAffT: data to be aug --> shape ()
    # yAffT: label to be aug --> shape ()
    # N_aug: number of aug --> int
    
        
    xAffTAug = []
    yAffTAug = []
    for eachAug in range(N_aug):
        for eachX, eachY in zip(xAffT, yAffT):
            
            
            if ards=='SYNTH':
                sx=np.random.uniform(0.5,2.0)
                sy=np.random.uniform(0.5,2.0)
                tx=np.random.uniform(-15,15)
                ty=np.random.uniform(-15,15)
                sh_ang=np.random.uniform(-np.pi/5,np.pi/5)
                rot_ang=np.random.uniform(-40,40)
            elif ards=='AFFNISTb_out':
                sx=np.random.uniform(0.8,1.2)
                sy=np.random.uniform(0.8,1.2)
                tx=np.random.uniform(-5,5)
                ty=np.random.uniform(-5,5)
                sh_ang=np.random.uniform(-np.pi/20,np.pi/20)
                rot_ang=np.random.uniform(-10,10)
            else:
                sx=np.random.uniform(0.5,2.0)
                sy=np.random.uniform(0.5,2.0)
                tx=np.random.uniform(-15,15)
                ty=np.random.uniform(-15,15)
                sh_ang=np.random.uniform(-np.pi/10,np.pi/10)
                rot_ang=np.random.uniform(-20,20)
            
            
            eachXPadTform=eachX+1
            
            szs=2*int(np.round(img_size/2))

            #eachXPadTform = np.pad(eachXPadTform, ((0,szs),(0,szs)), 'constant',constant_values=-1)
            eachXPadTform = np.pad(eachXPadTform, ((0,szs),(0,szs)), 'constant')
            
            tform_s = transform.AffineTransform(scale=(sx, sy), shear=sh_ang)
            eachXPadTform = transform.warp(eachXPadTform, tform_s.inverse)  
            
            cent=szs
            cy1=sx*cent/2; cx1=sy*cent/2;
            
            tform_t1 = transform.AffineTransform(translation=(cent-cy1, cent-cx1)) 
            eachXPadTform = transform.warp(eachXPadTform, tform_t1.inverse) 
            
            eachXPadTform = transform.rotate(eachXPadTform,rot_ang)
            
            #eachXPadTform = np.pad(eachXPadTform, ((15,15),(15,15)), 'constant',constant_values=-1)
            eachXPadTform = np.pad(eachXPadTform, ((15,15),(15,15)), 'constant')
            
            tform_t = transform.AffineTransform(translation=(tx, ty)) 
            eachXPadTform = transform.warp(eachXPadTform, tform_t.inverse)  
            eachXPadTform=eachXPadTform-1

            xAffTAug.append(eachXPadTform)
            yAffTAug.append(eachY)

    xAffTAug = np.asarray(xAffTAug)
    yAffTAug = np.asarray(yAffTAug) 

    return xAffTAug, yAffTAug

def new_index_matrix(max_index, n_samples_perclass, num_classes, repeat, y_train):
    seed = int('{}{}{}'.format(n_samples_perclass, num_classes, repeat))
    np.random.seed(seed)
    index = np.zeros([num_classes, n_samples_perclass], dtype=np.int64)
    for classidx in range(num_classes):
        max_samples = (y_train == classidx).sum()
        index[classidx] = np.random.randint(0, max_samples, (n_samples_perclass))
    return index


def resize(X, target_size):
    # Assume batch of grayscale images
    assert len(X.shape) == 3
    if target_size == X.shape[1]:
        return X
    X_resize = []
    for i in range(X.shape[0]):
        im = Image.fromarray(X[i])
        im_resize = im.resize((target_size, target_size))
        X_resize.append(np.asarray(im_resize))
    X_resize = np.stack(X_resize, axis=0)
    assert X_resize.shape[0] == X.shape[0]
    return X_resize


def take_samples(data, labels, index, num_classes):
    assert data.shape[0] == labels.shape[0]
    assert index.shape[0] == num_classes
    indexed_data = []
    new_labels = []
    for i in range(num_classes):
       class_data, class_labels = data[labels == i], labels[labels == i]
       indexed_data.append(class_data[index[i]])
       new_labels.append(class_labels[index[i]])
    return np.concatenate(indexed_data), np.concatenate(new_labels)


def load_data(dataset, num_classes, datadir='../data'):
    cache_file = os.path.join(datadir, dataset, 'dataset.hdf5')
    if os.path.exists(cache_file):
        with h5py.File(cache_file, 'r') as f:
            x_train, y_train = f['x_train'][()], f['y_train'][()]
            x_test, y_test = f['x_test'][()], f['y_test'][()]
            print('loaded from cache file data: x_train {} x_test {}'.format(x_train.shape, x_test.shape))
            return (x_train, y_train), (x_test, y_test)

    print('loading data from mat files')
    x_train, y_train, x_test, y_test = [], [], [], []
    for split in ['training', 'testing']:
        for classidx in range(num_classes):
            datafile = os.path.join(datadir, dataset, '{}/dataORG_{}.mat'.format(split, classidx))
            # loadmat(datafile)['xxO'] is of shape (H, W, N)
            data = loadmat(datafile)['xxO'].transpose([2, 0, 1]) # transpose to (N, H, W)
            label = np.zeros(data.shape[0], dtype=np.int64)+classidx
            print('split {} class {} data.shape {}'.format(split, classidx, data.shape))
            if split == 'training':
                x_train.append(data)
                y_train.append(label)
            else:
                x_test.append(data)
                y_test.append(label)
    # min_samples = min([x.shape[0] for x in x_train])
    # x_train = [x[:min_samples] for x in x_train]
    # y_train = [y[:min_samples] for y in y_train]
    x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)
    x_test, y_test = np.concatenate(x_test), np.concatenate(y_test)
    print('x_train.shape {} x_test.shape {}'.format(x_train.shape, x_test.shape))

    x_train = x_train / x_train.max(axis=(1, 2), keepdims=True)
    x_test = x_test / x_test.max(axis=(1, 2), keepdims=True)

    x_train = (x_train * 255.).astype(np.uint8)
    x_test = (x_test * 255.).astype(np.uint8)

    with h5py.File(cache_file, 'w') as f:
        f.create_dataset('x_train', data=x_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('x_test', data=x_test)
        f.create_dataset('y_test', data=y_test)
        print('saved to {}'.format(cache_file))

    return (x_train, y_train), (x_test, y_test)


def load_data_3D(dataset, num_classes):
    (x_train, y_train), (x_test, y_test) = load_data(dataset, num_classes)
    # Convert to 1 channel grayscale
    x_train = x_train.reshape(-1, 1, x_train.shape[1], x_train.shape[2])
    # Convert to 3 channels by replicating
    x_train = np.repeat(x_train, axis=1, repeats=3)

    x_test = x_test.reshape(-1, 1, x_test.shape[1], x_test.shape[2])
    x_test = np.repeat(x_test, axis=1, repeats=3)
    return (x_train, y_train), (x_test, y_test)

def take_train_samples(x_train, y_train, n_samples_perclass, num_classes, repeat):
    max_index = x_train.shape[0] // num_classes
    train_index = new_index_matrix(max_index, n_samples_perclass, num_classes, repeat, y_train)
    x_train_sub, y_train_sub = take_samples(x_train, y_train, train_index, num_classes)
    return x_train_sub, y_train_sub

def take_train_val_samples(x_train, y_train, n_samples_perclass, num_classes, repeat):
    max_index = x_train.shape[0]//num_classes
    train_index = new_index_matrix(max_index, n_samples_perclass, num_classes, repeat, y_train)

    val_samples = n_samples_perclass // 10 # Use 10% for validation

    if val_samples >= 1:
        val_index = train_index[:, :val_samples]
        x_val, y_val = take_samples(x_train, y_train, val_index, num_classes)
        assert x_val.shape[0] == y_val.shape[0]
        print('validation data shape {}'.format(x_val.shape), end=' ')
    else:
        x_val, y_val = None, None
        print('validation data {}'.format(x_val), end=' ')

    train_sub_index = train_index[:, val_samples:]
    x_train_sub, y_train_sub = take_samples(x_train, y_train, train_sub_index, num_classes)
    print('train data shape {}'.format(x_train_sub.shape))

    if x_val is not None:
        assert x_val.shape[0] + x_train_sub.shape[0] == n_samples_perclass*num_classes
    else:
        assert x_train_sub.shape[0] == n_samples_perclass*num_classes


    return (x_train_sub, y_train_sub), (x_val, y_val)



def dataset_config(dataset):
    assert dataset in ['toy', 'SYNTH', 'MNIST', 'OAM_t10', 'OAM', 'AFFNISTb','AFFNISTb_out','OMNIGLOT','CHAND','THAND']
    if dataset in ['toy']:
        rm_edge = True
        num_classes = 3
        img_size = 32
        #po_train_max = 4  # maximum train samples = 2^po_max
        po_train = [1,2,4,6,8,10]
    elif dataset in ['SYNTH']:
        rm_edge = True
        num_classes = 10
        #po_train_max = 1  # maximum train samples = 2^po_max
        po_train = [1,2]
        img_size = 100
    elif dataset in ['MNIST']:
        rm_edge = True
        num_classes = 10
        #po_train_max = 4  # maximum train samples = 2^po_max
        po_train = [1,2,4,6,8,10]
        img_size = 28
    elif dataset in ['OAM_t10']:
        rm_edge = True
        num_classes = 32
        #po_train_max = 4  # maximum train samples = 2^po_max
        po_train = [1,2,4,6,8,10]
        img_size = 151
    elif dataset in ['OAM']:
        rm_edge = True
        num_classes = 32
        # po_train_max = 4  # maximum train samples = 2^po_max
        po_train = [1,2,4,6,8,10]
        img_size = 151
    elif dataset in ['AFFNISTb']:
        rm_edge = True
        num_classes = 10
        img_size = 64
        # po_train_max = 4  # maximum train samples = 2^po_max
        po_train = [1,2,4,6,8,10]
    elif dataset in ['AFFNISTb_out']:
        rm_edge = True
        num_classes = 10
        img_size = 64
        # po_train_max = 4  # maximum train samples = 2^po_max
        po_train = [1,2,4,6,8,10]
    elif dataset in ['OMNIGLOT']:
        rm_edge = True
        num_classes = 30
        img_size = 105
        # po_train_max = 4  # maximum train samples = 2^po_max
        po_train = [1,2,4,6,8]
    elif dataset in ['CHAND']:
        rm_edge = True
        num_classes = 15
        img_size = 64
        # po_train_max = 4  # maximum train samples = 2^po_max
        po_train = [1,2,4,6,8,10]
    elif dataset in ['THAND']:
        rm_edge = True
        num_classes = 9
        img_size = 50
        # po_train_max = 4  # maximum train samples = 2^po_max
        po_train = [1,2,4,6,8,10]


    



    return num_classes, img_size, po_train, rm_edge
