import os
import pickle
from functools import lru_cache
from datetime import datetime
from torchvision.utils import make_grid
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR
from IPython.core.debugger import set_trace
import torchvision.transforms as transforms
from PIL import Image
from utils import *
from model import MNISTNet
from sklearn.metrics import accuracy_score
# from cifar_models import resnet18


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--model', default='shallowcnn', type=str, choices=['vgg11', 'shallowcnn', 'resnet18'])
parser.add_argument('--plot', action='store_true')
parser.add_argument('--naug', default=1, type=int)
parser.add_argument('--flops', action='store_true')
args = parser.parse_args()

#if args.dataset == 'MNIST':
#    assert args.model not in ['vgg11']

num_classes, img_size, po_train_max, _ = dataset_config(args.dataset)

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.flops:
    device = 'cpu'
    from pypapi import events, papi_high as high

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data_3D(args.dataset, num_classes)
    print('loaded data, x_train.shape {}, x_test.shape {}'.format(x_train.shape, x_test.shape))

    # x_train shape: (class_idx*n_samples_perclass, args.img_size, args.img_size)
    x_train = (x_train.astype(np.float32) / 255. - 0.5) / 0.5
    x_test = (x_test.astype(np.float32) / 255. - 0.5) / 0.5

    if args.model == 'vgg11':
        model = models.vgg11_bn(num_classes=num_classes).to(device)
    elif args.model == 'shallowcnn':
        model = MNISTNet(3, num_classes, 2*(img_size+15)).to(device)
    if args.model == 'resnet18':
        model = models.resnet18(num_classes=num_classes).to(device)
    torch.save(model.state_dict(), './model_init.pth')

    if args.flops:
        model = model.double()
        model.eval()
        with torch.no_grad():
            high.start_counters([events.PAPI_DP_OPS,])
            x_test_batch = torch.rand(1, 3, img_size, img_size, dtype=torch.float64)
            test_logit = model(x_test_batch)
            test_gflops =high.stop_counters()[0] / 1e9
            print('test gflops: {}'.format(test_gflops))
        model.train()

    accs = []
    all_preds = []
    num_repeats = 1
    AUG_N = args.naug
    for n_samples_perclass in [2**i for i in range(0, po_train_max+1)]:
    # for n_samples_perclass in [512]:
        for repeat in range(num_repeats):
            model.load_state_dict(torch.load('./model_init.pth'))
            (x_train_sub_beforeAug, y_train_sub_beforeAug), (x_valbeforePad, y_valbeforePad) = take_train_val_samples(x_train, y_train, n_samples_perclass, num_classes, repeat)
            

            x_train_sub_beforeAugReshape = rearrange(x_train_sub_beforeAug, 'b c w h -> b w h c') 
            np.random.seed(seed=None)
            # x_train_sub_breforeReshape, y_train_sub_aug = take_affine_tform_aug(x_train_sub_beforeAugReshape, y_train_sub_beforeAug,img_size,args.dataset, N_aug=AUG_N) #
            x_train_sub_breforeReshape, y_train_sub_aug = take_affine_tform_augV2(x_train_sub_beforeAugReshape, y_train_sub_beforeAug,img_size,args.dataset, N_aug=AUG_N)
            np.random.seed(0)
            x_train_sub_aug = rearrange(x_train_sub_breforeReshape, 'b w h c -> b c w h') 
            x_train_sub_beforeAugPad = []
            for eachX_trainBeforePad in x_train_sub_beforeAugReshape:
                x_train_sub_beforeAugPad.append(np.pad(eachX_trainBeforePad, ((15+int(np.round(img_size/2)),15+int(np.round(img_size/2))),(15+int(np.round(img_size/2)),15+int(np.round(img_size/2))),(0,0)), 'constant', constant_values=-1)) # pad the original image ((top, bottom), (left, right)) https://stackoverflow.com/questions/38191855/zero-pad-numpy-array)
            x_train_original = rearrange(np.asarray(x_train_sub_beforeAugPad), 'b w h c -> b c w h') 
            x_train_sub = np.concatenate((x_train_original, x_train_sub_aug), axis=0)
            y_train_sub = np.concatenate((y_train_sub_beforeAug, y_train_sub_aug), axis=0)
            if x_valbeforePad is None:
                x_val = None
                y_val = None
            else:
                x_val_Pad = []
                x_val_beforePad = rearrange(x_valbeforePad, 'b c w h -> b w h c') 
                for eachVal in x_val_beforePad:
                    x_val_Pad.append(np.pad(eachVal, ((15+int(np.round(img_size/2)),15+int(np.round(img_size/2))),(15+int(np.round(img_size/2)),15+int(np.round(img_size/2))),(0,0)), 'constant', constant_values=-1)) # pad the original image ((top, bottom), (left, right)) https://stackoverflow.com/questions/38191855/zero-pad-numpy-array) 
                x_val_PadReshape = rearrange(np.asarray(x_val_Pad), 'b w h c -> b c w h') 
                x_val = x_val_PadReshape.copy()
                y_val = y_valbeforePad.copy()
            
            
            x_val_shape = 0 if x_val is None else x_val.shape
            print('============== perclass samples {} repeat {} x_train_sub.shape {} x_val.shape {} ============'.format(n_samples_perclass, repeat, x_train_sub.shape, x_val_shape))

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=5e-4)

            save_path = 'results-new-validation/{}/samples-{}-model-{}/'.format(args.dataset, n_samples_perclass, type(model).__name__)
            Path(save_path).mkdir(parents=True, exist_ok=True)
            ckpt_path = os.path.join(save_path, 'repeat{}.pkl'.format(repeat))
            best_val_acc = 0.0

            for epoch in range(args.epochs):
                perm = np.random.permutation(x_train_sub.shape[0])
                x_train_sub_perm = x_train_sub[perm]
                y_train_sub_perm = y_train_sub[perm]

                if args.plot:
                    fig, axes = plt.subplots(ncols=num_classes, nrows=1)
                    for k in range(num_classes):
                        class_data = x_train_sub_perm[y_train_sub_perm == k][:64]
                        class_data = class_data.reshape(class_data.shape[0], 3, *class_data.shape[2:])
                        print(class_data.shape, class_data.dtype, class_data.min(), class_data.max())
                        axes[k].imshow(make_grid(torch.from_numpy(class_data), nrow=16, pad_value=1).permute(1, 2, 0))
                        axes[k].set_axis_off()
                    plt.savefig('samples.pdf', dpi=400)
                    plt.show()

                # train
                model.train()
                for i in range(0, x_train_sub_perm.shape[0], args.batch_size):
                    inputs = x_train_sub_perm[i: i + args.batch_size]
                    targets = y_train_sub_perm[i: i + args.batch_size]
                    inputs = torch.from_numpy(inputs).to(device)
                    targets = torch.from_numpy(targets).to(device)
                    # if inputs.shape[0] != args.batch_size:
                    #     break

                    optimizer.zero_grad()
                    outputs = model(inputs.type(torch.cuda.FloatTensor))
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    _, max_indices = torch.max(outputs, 1)

                    train_acc = (max_indices == targets).type(torch.float).mean()
                    if (i//args.batch_size) % 10 == 0:
                        print('epoch {} iter {} train loss {:.5f} acc {:.5f}'.format(epoch, i//args.batch_size, loss.item(), train_acc))

                # validation
                if x_val is not None:
                    model.eval()
                    with torch.no_grad():
                        val_logits = []
                        for i in range(0, x_val.shape[0], 100):
                            x_val_batch = torch.from_numpy(x_val[i:i+100]).to(device)
                            batch_logit = model(x_val_batch)
                            val_logits.append(batch_logit.cpu().numpy())
                        val_logits = np.concatenate(val_logits)
                        val_acc = (np.argmax(val_logits, axis=1) == y_val).mean()
                        print('epoch {} val acc {:.5f}'.format(epoch, val_acc))
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            state = dict(model=model.state_dict(), best_val_acc=val_acc, epoch=epoch)
                            torch.save(state, ckpt_path)
                            print('saved to ' + ckpt_path)
                else:
                    state = dict(model=model.state_dict(), best_val_acc=-1, epoch=epoch)
                    torch.save(state, ckpt_path)
                    print('saved to ' + ckpt_path)
            # test
            model.eval()
            with torch.no_grad():
                state = torch.load(ckpt_path)
                model.load_state_dict(state['model'])
                print('recovered from {}'.format(ckpt_path))
                print('samples {} repeat {} best val acc {}, epoch {}'.format(n_samples_perclass, repeat, state['best_val_acc'],
                                                                           state['epoch']), end=' ')
                logit = []
                for i in range(0, x_test.shape[0], 100):
                    x_test_batchBeforePAD = x_test[i:i+100] # torch.from_numpy(x_test[i:i+100]).to(device)
                    
                    x_test_batchRearrange = rearrange(x_test_batchBeforePAD, 'b c w h -> b w h c') 
                    x_test_batchBeforeRearrange = [] 
                    for eachTest in x_test_batchRearrange:
                        x_test_batchBeforeRearrange.append(np.pad(eachTest, ((15+int(np.round(img_size/2)),15+int(np.round(img_size/2))),(15+int(np.round(img_size/2)),15+int(np.round(img_size/2))),(0,0)), 'constant', constant_values=-1)) # pad the original image ((top, bottom), (left, right)) https://stackoverflow.com/questions/38191855/zero-pad-numpy-array) 
                    x_test_batchNp = rearrange(np.asarray(x_test_batchBeforeRearrange), 'b w h c -> b c w h') 
                    x_test_batch = torch.from_numpy(x_test_batchNp).to(device)
                    
                    test_logit = model(x_test_batch)
                    logit.append(test_logit.cpu().numpy())
                  
                logit = np.concatenate(logit)
                y_pred = np.argmax(logit, axis=1)
                test_acc = (y_pred == y_test).mean()
                del state['model']
                state['test_acc'] = test_acc
                state['confusion_matrix'] = confusion_matrix(y_test, y_pred)
                print('test acc {:.5f}'.format(test_acc))
                print(state['confusion_matrix'])
                with open(ckpt_path, 'wb') as f:
                    pickle.dump(state, f)
                print('saved to {}'.format(ckpt_path))
                accs.append(accuracy_score(y_test, y_pred))
                all_preds.append(y_pred)
              

    accs = np.array(accs).reshape(-1, num_repeats)
    preds = np.stack(all_preds, axis=0)
    preds = preds.reshape([preds.shape[0] // num_repeats, num_repeats, preds.shape[1]])

    results_dir = 'results/final/{}/'.format(args.dataset)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    result_file = os.path.join(results_dir, 'nn_aug'+str(AUG_N)+'_{}.hdf5'.format(args.model))
    with h5py.File(result_file, 'w') as f:
        f.create_dataset('accs', data=accs)
        f.create_dataset('preds', data=preds)
        f.create_dataset('y_test', data=y_test)
    print('saved to ' + result_file)
