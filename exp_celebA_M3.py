# 用Dtrain(181952)训练出多分类器，利用Dtest算出准确度，得到αtrain
# 用Dsample训练多分类器，利用Dtest算出准确度，得到αsample

# python multi_categorical_gans/metrics/mse_predictions_by_categorical.py \
#     --data_format_x=sparse --data_format_y=dense --data_format_test=sparse \
#     data/synthetic/mix_small/synthetic-train.features.npz \
#     samples/arae/synthetic/mix_small/sample.features.npy \
#     data/synthetic/mix_small/synthetic-test.features.npz \
#     data/synthetic/mix_small/metadata.json