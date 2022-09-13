
def get_save_name(version, base_data_folder_name, checkpoint_name):
    return f'{version}_{base_data_folder_name}_{checkpoint_name}'

def get_results_folder_name(data_folder, start_no, end_no, suffix):
    return f'{data_folder}_p{start_no}to{end_no}_{suffix}'
