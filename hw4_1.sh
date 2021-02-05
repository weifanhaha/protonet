# python3 p1.py $1 $2 $3 $4

python p1/predict.py --load p1/best_model.pth --test_csv $1 --test_data_dir $2 \
--testcase_csv $3 --output_csv $4

# python p1/predict.py --load p1/best_model.pth --test_csv ./hw4_data/val.csv --test_data_dir ./hw4_data/val \
# --testcase_csv ./hw4_data/val_testcase.csv --output_csv ./p1/output.csv

# ./hw4_1.sh ./hw4_data/val.csv ./hw4_data/val ./hw4_data/val_testcase.csv ./p1/output.csv
