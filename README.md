# Few-Shot Classification with Prototypical Network and Improved with Data Hallucination

Implemented [Prototypical Network](https://arxiv.org/pdf/1703.05175.pdf) for image classfication task on Mini-ImageNet in a few-shot learning manner. A hallucinator model for [data hallucination](https://arxiv.org/pdf/1801.05401.pdf) is also included to improve the performance of ProtoNet.

## Usage

### Download Dataset

```
./get_dataset.sh
```

### Install packages

```
pip install -r requirements.txt
```

### Train

To train ProtoNet with / without data hallucination, please run the command below and please take care of the path of training data and output model path.

Train ProtoNet

```
python train.py
```

Train ProtoNet with data hallucination

```
python train_hallucination.py
```

### Predict Labels

You can predict the labels of the images in the test dataset with trained models.

Predict labels with ProtoNet

```
python predict.py --load $model_path --test_csv $test_csv \
    --test_data_dir $test_data_dir --testcase_csv $testcase_csv \
    --output_csv $output_csv
```

Predict labels with ProtoNet trained with hallucinator

```
python predict_hallucination.py --load $model_path --test_csv $test_csv \
    --test_data_dir $test_data_dir --testcase_csv $testcase_csv \
    --output_csv $output_csv
```

-   `$model_path`: path of the trained model
-   `$test_csv`: path of the test csv file
-   `$test_data_dir`: path of the test images directory
-   `$testcase_csv`: path of test case on test set
-   `$output_csv`: path of the output csv

## Reference

The implementation of ProtoNet refers to [Prototypical-Networks-for-Few-shot-Learning-PyTorch github](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch)
