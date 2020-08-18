import tensorflow as tf
from tensorflow.keras import layers
from layers import AUGRU
from activations import Dice
from model import DIEN
import pandas as pd
import alibaba_data_reader as data_reader

def main():
    train_data, test_data, embedding_count = data_reader.get_data()
    embedding_features_list = data_reader.get_embedding_features_list()
    user_behavior_features = data_reader.get_user_behavior_features()
    embedding_count_dict = data_reader.get_embedding_count_dict(embedding_features_list, embedding_count)
    embedding_dim_dict = data_reader.get_embedding_dim_dict(embedding_features_list)
    model = DIEN(embedding_count_dict, embedding_dim_dict, embedding_features_list, user_behavior_features)
    min_batch = 0
    batch = 100
    label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level, hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show, min_batch = data_reader.get_batch_data(train_data, min_batch, batch = batch)
    print(label)
    #print(tf.concat([hist_brand_behavior_clk, hist_cate_behavior_clk], -1))



if __name__ == "__main__":
    print("GPU Available: ", tf.test.is_gpu_available())
    main()