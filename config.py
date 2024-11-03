from os.path import join

class FLAGES(object):


    lr = 4e-4
    decay_rate = 0.99
    decay_step = 10000
    num_workers = 8
    train_batch_size = 4
    test_batch_size = 1
    val_batch_size = 1

    trainA_path = r"D:\Desk\PycharmProject\lightweight-uie\euvp-lightweight\myfive-k7-c7\datasets\train_data\input/"  #raw
    trainB_path = r"D:\Desk\PycharmProject\lightweight-uie\euvp-lightweight\myfive-k7-c7\datasets\train_data\target/"  #reference

    valA_path = r"D:\Desk\PycharmProject\lightweight-uie\euvp-lightweight\myfive-k7-c7\datasets\val_data\input/"  #raw
    valB_path = r"D:\Desk\PycharmProject\lightweight-uie\euvp-lightweight\myfive-k7-c7\datasets\val_data\target/"  #reference
    testA_path = r"D:\Desk\PycharmProject\lightweight-uie\euvp-lightweight\myfive-k7-c7\datasets\val_data\input/"  #raw

    out_path = r"D:\Desk\PycharmProject\lightweight-uie\euvp-lightweight\myfive-k7-c7\results\output_test/"  #output

    model_dir = r'D:\Desk\PycharmProject\lightweight-uie\euvp-lightweight\myfive-k7-c7\results\model_save/'

    backup_model_dir = join(model_dir)