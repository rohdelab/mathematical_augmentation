

# Use different models to do classification on different datasets


python CNNc.py --dataset SYNTH --model shallowcnn --naug 100
python CNNc.py --dataset SYNTH --model vgg11 --naug 1
python CNNc.py --dataset SYNTH --model vgg11 --naug 10
python CNNc.py --dataset MNIST --model vgg11 --naug 100
python CNNc.py --dataset SYNTH --model vgg11 --naug 100
python CNNc.py --dataset AFFNISTb_out --model shallowcnn --naug 100
python CNNc.py --dataset AFFNISTb_out --model vgg11 --naug 1
python CNNc.py --dataset AFFNISTb_out --model vgg11 --naug 10
python CNNc.py --dataset AFFNISTb --model vgg11 --naug 100
python CNNc.py --dataset AFFNISTb_out --model vgg11 --naug 100
python CNNc.py --dataset OMNIGLOT --model resnet18 --naug 100
python CNNc.py --dataset OMNIGLOT --model shallowcnn --naug 100
python CNNc.py --dataset OMNIGLOT --model vgg11 --naug 100