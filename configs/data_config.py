megadepth_root_dir = '/data2/MegaDepth_v1'
megadepth_train_val_npz_root = f"{megadepth_root_dir}/megadepth_indices/scene_info_0.1_0.7"
megadepth_train_list = f'{megadepth_root_dir}/megadepth_indices/trainvaltest_list/train_list.txt'
megadpeth_val_list = f'{megadepth_root_dir}/megadepth_indices/trainvaltest_list/val_list.txt'
megadepth_test_npz_root = 'assets/megadepth_test_1500_scene_info'
megadepth_test_list = 'assets/megadepth_test_1500_scene_info/megadepth_test_1500.txt'



scannet_root_dir = '/data2/scannet'
scannet_train_root_dir = f'{scannet_root_dir}/scannet_all'
scannet_train_val_npz_root = f'{scannet_root_dir}/scannet_indices/scene_data/train'
scannet_train_list = f'{scannet_root_dir}/scannet_indices/scene_data/train_list/scannet_all.txt'
scannet_train_intrinsic_path = f'{scannet_root_dir}/scannet_indices/intrinsics.npz'

scannet_test_root_dir = f'{scannet_root_dir}/scannet_test_1500'
scannet_test_npz_root = 'assets/scannet_test_1500'
scannet_test_list = 'assets/scannet_test_1500/scannet_test.txt'
scannet_test_intrinsic_path = 'assets/scannet_test_1500/intrinsics.npz'
