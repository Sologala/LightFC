class EnvironmentSettings:
    def __init__(self, env_num=0):
        self.workspace_dir = r"/home/wen/ws/LightFC/output/"  # Base directory for saving network checkpoints.
        self.tensorboard_dir = r"/home/wen/ws/LightFC/output/"  # Directory for tensorboard files.
        self.pretrained_networks = r"/home/wen/ws/LightFC/pretraind_models/"

        self.lasot_dir = ''
        self.got10k_dir = '/home/wen/ws/LightFC/data/got10k/train_data/'
        self.got10k_val_dir = '/home/wen/ws/LightFC/data/got10k/val/'
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_dir = ''
        self.trackingnet_lmdb_dir = ''
        self.coco_dir = ''
        self.coco_lmdb_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''

        self.imagenet_dir = ''
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
