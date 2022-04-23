

# Use different models to do classification on different datasets


python CNNc.py --dataset MNIST --model resnet18 --naug 50
python CNNc.py --dataset MNIST --model vgg11 --naug 50
python CNNc.py --dataset MNIST --model shallowcnn --naug 50
python TRADc.py --dataset MNIST --feature raw --classifier KNN --naug 50
python RCDTNS.py --dataset MNIST
python RCDTNS_0.py --dataset MNIST


python CNNc.py --dataset AFFNISTb --model resnet18 --naug 50
python CNNc.py --dataset AFFNISTb --model vgg11 --naug 50
python CNNc.py --dataset AFFNISTb --model shallowcnn --naug 50
python TRADc.py --dataset AFFNISTb --feature raw --classifier KNN --naug 50
python RCDTNS.py --dataset AFFNISTb
python RCDTNS_0.py --dataset AFFNISTb

python CNNc.py --dataset OMNIGLOT --model resnet18 --naug 50
python CNNc.py --dataset OMNIGLOT --model vgg11 --naug 50
python CNNc.py --dataset OMNIGLOT --model shallowcnn --naug 50
python TRADc.py --dataset OMNIGLOT --feature raw --classifier KNN --naug 50
python RCDTNS.py --dataset OMNIGLOT
python RCDTNS_0.py --dataset OMNIGLOT




