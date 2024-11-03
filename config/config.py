from os.path import join

class FLAGES(object):

    lr = 4e-4
    decay_rate = 0.99
    decay_step = 10000
    num_workers = 8
    train_batch_size = 4
    test_batch_size = 1
    val_batch_size = 1

    # UIEB path
    trainA_path = r"./input"  #raw
    trainB_path = r"./target"  #reference

    valA_path = r"./input"  #raw
    valB_path = r"./target"  #reference
    testA_path = r"./raw"  #raw
    out_path = r"./output_test"  #output
    model_dir = r'./model_save'

    backup_model_dir = join(model_dir)
