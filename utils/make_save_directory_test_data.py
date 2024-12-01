import datetime
from os import path, mkdir

def make_save_directory_test_data(cfg):
    today = datetime.datetime.today().strftime('%Y%m%d')
    model_save_dir = path.join(cfg.directory.test_dir, today + '_' + cfg.model.exp_name + '_' + str(cfg.model.exp_no))
    if not path.exists(model_save_dir):
        mkdir(model_save_dir)
    
    """output_dir = path.join(cfg.directory.output_dir, today + '_' + cfg.model.exp_name + '_' + str(cfg.model.exp_no))
    if not path.exists(output_dir):
        mkdir(output_dir)"""
    
    return model_save_dir