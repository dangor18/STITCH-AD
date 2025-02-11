MODE = 1             # 1: train, 2: test, 3: eval
MASK_TYPE = 3             # 1: random block, 2: half, 3: external, 4: (external, random block), 5: (external, random block, half)
SEED = 10            # random seed 10
GPU = [0]            # list of gpu ids
DEBUG = 0            # turns on debugging mode
VERBOSE = 0          # turns on verbose mode in the output console

dataroot = 'C:\Development\STITCH-AD\PRE_PROCESS\DATA\PATCHES\DRGB_80_172'
workers = 4
normal_class = 'good'
VAL_SPLIT = 0.25
MODIFIED = False

TRAIN_MASK_FLIST = 'C:\Development\STITCH-AD\PATCH_AD\SCADN\mask'
TEST_MASK_FLIST = 'C:\Development\STITCH-AD\PATCH_AD\SCADN\mask'


LR = 0.0001                    # learning rate
D2G_LR = 0.1                   # discriminator/generator learning rate ratio
BETA1 = 0.0                    # adam optimizer beta1
BETA2 = 0.9                    # adam optimizer beta2

USE_LR_DECAY = False

BATCH_SIZE = 64                 # input batch size for training
INPUT_SIZE = 256               # input image size for training 0 for original size
INPUT_CHANNELS = 4
SCALES = [1, 2, 3]              # list of what mask scales to use
                                # 0: half-width strips, 1: quarter-width strips, 2: eighth-width strips, 3: sixteenth-width strips
MAX_EPOCHS = 200                # maximum number of iterations to train the model

REC_LOSS_WEIGHT = 1             # l1 loss weight
FM_LOSS_WEIGHT = 0           # feature-matching loss weight
INPAINT_ADV_LOSS_WEIGHT = 0.001  # adversarial loss weight

GAN_LOSS = 'nsgan'               # nsgan | lsgan | hinge

LOG_INTERVAL = 1            # how many iterations to wait before logging training status (0: never)

STAGE = [1]
DATASET = 'STITCH-O'
SUB_SET = ''
PATH = 'C:\Development\STITCH-AD\PATCH_AD\SCADN\checkpoints'
DEBUG = 0