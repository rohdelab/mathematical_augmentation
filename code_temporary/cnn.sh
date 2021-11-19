

# Use different models to do classification on different datasets


python CNNc.py --dataset MNIST --model vgg11 --naug 50
python CNNc.py --dataset SYNTH --model vgg11 --naug 50
python CNNc.py --dataset AFFNISTb --model vgg11 --naug 50
python CNNc.py --dataset OMNIGLOT --model vgg11 --naug 50
python CNNc.py --dataset AFFNISTb_out --model vgg11 --naug 50

python CNNc.py --dataset AFFNISTb_out --model vgg11 --naug 25
python CNNc.py --dataset SYNTH --model vgg11 --naug 25
python CNNc.py --dataset OMNIGLOT --model vgg11 --naug 25
python CNNc.py --dataset MNIST --model vgg11 --naug 25
python CNNc.py --dataset AFFNISTb --model vgg11 --naug 25



