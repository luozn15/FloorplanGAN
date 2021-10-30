from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 8
_C.SYSTEM.NUM_WORKERS = 4
_C.SYSTEM.HOSTNAMES = []

_C.DATASET = CN()
_C.DATASET.NAME = "rplan"
_C.DATASET.SUBSET = ""
_C.DATASET.BATCHSIZE = 128

_C.MODEL = CN()
_C.MODEL.GENERATOR = CN()
_C.MODEL.RENDERER = CN()
_C.MODEL.RENDERER.RENDERING_SIZE = 64
_C.MODEL.DISCRIMINATOR = CN()

_C.TRAIN = CN()
_C.TRAIN.LEARNING_RATE = 0.00002
_C.TRAIN.NUM_EPOCHS = 3000

_C.TENSORBOARD = CN()
_C.TENSORBOARD.SAVE_INTERVAL = 100

_C.PATH = CN()
_C.PATH.RPLAN = []
_C.PATH.Z_FILE = "fixed_z/fixed_xyaw_rplan_0320.pkl"
_C.PATH.LOG_DIR = "runs_rplan"

_C.MANUAL = CN()


def get_cfg():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
