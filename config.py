from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 8
_C.SYSTEM.NUM_WORKERS = 4

_C.DATASET = CN()
_C.DATASET.NAME = "rplan"
_C.DATASET.SUBSET = "names_0-1_1-1_2-1_3-1_4-0_5-0_6-0_7-1_8-0_9-1.pkl"
_C.DATASET.BATCHSIZE = 128

_C.TRAIN = CN()
_C.TRAIN.LEARNING_RATE = 0.00002
_C.TRAIN.NUM_EPOCHS = 3000

_C.TENSORBOARD = CN()
_C.TENSORBOARD.SAVE_INTERVAL = 300

_C.PATH = CN()
_C.PATH.Z_FILE = "fixed_z/fixed_xyaw_rplan_0320.pkl"
_C.LOG_DIR = "runs_rplan"


def get_cfg():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
