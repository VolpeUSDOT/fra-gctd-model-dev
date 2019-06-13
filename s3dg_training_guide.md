s3dg Model Development Guide

1. Data Set Construction

	TODO

2. Model Training and Evaluation

The script s3dg_estimator.py can be used to train, evaluate and make predictions using s3dg models, as well as to export models to the (new) SaveModel format. Training and evaluation can be performed jointly or separately.

The training and evaluation procedure presented in the guide proceeds in (up to) three phases:

	1) Initializing one model's feature extraction layers using parameters pre-trained on Kinetics 600, initializing its logits (i.e. output) layer randomly and only training its logits layer. The primary reason for initializing the logits layer randomly is that the number of neurons in that layer during this training will differ from the original number of layers in the pre-training if the number of classes in the classification problems differ. The learning rate is typically set to 0.1, which is considered to be high. This practice is known as transfer learning. A checkpoint directory path is to be specified (as demonstrated below) for the script to use to periodically write the state of the model and its performance metrics to disk during training.

	2) Initializing a second model using all of the layers from the first, and then training all of its layers. The learning rate is typically reduced to 0.001 with the objective of encouraging generalizeability and preventing overfitting by preserving the "knowledge" learned from the much larger and more diverse transfer data set. This step in transfer learning is called fine-tuning. This phase is technically optional, but we perform it because it has historically yielded better prediction performance. Because of an outstanding bug in the Estimator API of TensorFlow, a second separate checkpoint directory path must be specified when performing this step.

	3) Resuming training of the second model but with a reduced learning rate (e.g. 0.00001). Learning rate reduction can help the optimization algorithm settle into a local minimum and eek out the last bit of prediction performance. While technically the original checkpoint path of the second model can be reused, specifying a third separate directory helps to organize checkpoints and experiment with different hyperparameters.

During each phase, the tensorboard application can be used to monitor training using graphical visualizations of metric scalar values and per-layer parameter distributions.

Below are examples of three different invocations of s3dg_estimator.py that perform the three tasks described above:

	1) python s3dg_estimator.py --mode train_and_eval --model_dir C:/Users/Public/fra-gctd-project/Models/ramsey_nj/s3dg-estimator-pretrained-init --checkpoint_path C:/Users/Public/fra-gctd-project/Models/pre-trained/s3dg_kinetics_600_rgb/model.ckpt --train_subset_dir_path C:/Users/Public/fra-gctd-project/Data_Sets/ramsey_nj/20180419 --eval_subset_dir_path C:/Users/Public/fra-gctd-project/Data_Sets/ramsey_nj/20180420 --monitor_steps 532 --learning_rate 0.1 --batch_size 2 --variables_to_warm_start s3dg_convs --variables_to_train s3dg_logits

	*Note that the monitor_steps parameter, which governs when training metrics are logged to the console and to disk, is half of the number of training exampes because the batch size is 2, meaning two examples are processed per step. The choice to evaluate the model once per epoch is arbitrary. Evaluation can occur more frequently, but keep in mind that while evaluation is taking place training is stopped (so that the evaluator can use the same compute resources).

	2) python s3dg_estimator.py --mode train_and_eval --model_dir C:/Users/Public/fra-gctd-project/Models/ramsey_nj/s3dg-estimator-coarsetuned-init --checkpoint_path C:/Users/Public/fra-gctd-project/Models/ramsey_nj/s3dg-estimator-pretrained-init --train_subset_dir_path C:/Users/Public/fra-gctd-project/Data_Sets/ramsey_nj/20180419 --eval_subset_dir_path C:/Users/Public/fra-gctd-project/Data_Sets/ramsey_nj/20180420 --monitor_steps 1064 --learning_rate 0.001 --batch_size 1

	*Note that we do not need to explicitly specify variables_to_warm_start nor variables_to_train. We also reduced the batch size because performing backpropagation over the entire network rather than just the logits layer consumes significantly more GPU memory, leaving less space to store training examples. The monitor_steps value is reduced accordingly.

	3) python s3dg_estimator.py --mode train_and_eval --model_dir C:/Users/Public/fra-gctd-project/Models/ramsey_nj/s3dg-estimator-finetuned-init --checkpoint_path C:/Users/Public/fra-gctd-project/Models/ramsey_nj/s3dg-estimator-coarsetuned-init --train_subset_dir_path C:/Users/Public/fra-gctd-project/Data_Sets/ramsey_nj/20180419 --eval_subset_dir_path C:/Users/Public/fra-gctd-project/Data_Sets/ramsey_nj/20180420 --monitor_steps 1064 --learning_rate 0.00001 --batch_size 1

Tensorboard can be run using the following command: tensorboard --logdir C:/Users/Public/fra-gctd-project/Models/ramsey_nj. A URL hsould be printed to teh console that can be navigated to using a web browser.