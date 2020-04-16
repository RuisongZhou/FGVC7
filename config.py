
class Config():
    def __init__(self):
        ##################################################
        # Training Config
        ##################################################
        self.GPU = '0'                   # GPU
        self.workers = 4                 # number of Dataloader workers
        self.epochs = 40                # number of epochs
        self.batch_size = 16             # batch size
        self.learning_rate = 1e-3        # initial learning rate

        ##################################################
        # Model Config
        ##################################################
        self.image_size = (448, 448)     # size of training images
        self.net = 'inception_mixed_6e'  # feature extractor
        self.num_attentions = 32        # number of attention maps
        self.beta = 5e-2                 # param for update feature centers

        ##################################################
        # Dataset/Path Config
        ##################################################

        # saving directory of .ckpt models
        self.save_dir = './weights/'
        self.model_name = 'model_{}_{}.ckpt'.format(self.net, self.image_size[0])
        self.log_name = 'train.log'

        # checkpoint model for resume training
        self.ckpt = False
        #self.ckpt = self.save_dir + self.model_name

        ##################################################
        # Eval Config
        ##################################################
        self.visualize = False
        self.eval_ckpt = self.save_dir + self.model_name
        self.eval_savepath = './result/'


    def refresh(self):
        self.model_name = 'model_{}_{}.ckpt'.format(self.net, self.image_size[0])
        self.eval_ckpt = self.save_dir + self.model_name