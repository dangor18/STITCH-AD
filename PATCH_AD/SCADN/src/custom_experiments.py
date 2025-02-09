import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .models import InpaintingModel
from .utils import Progbar
from .metrics import PSNR
from .evaluate import evaluate
from .dataset import BlockMask
from.custom_dataset import load_data

class ExpStitchO():
    def __init__(self, config):
        self.config = config

        model_name = 'coarse'
        if 2 in config.STAGE:
            model_name = 'fine'

        self.debug = False
        self.model_name = model_name
        self.epoch = 0
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)
        self.scale_norm = [1 for _ in range(len(config.SCALES))]

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.mask_set = BlockMask(config)

        self.dataset_info, self.dataset = load_data(config)

        # self.dataset = {
        #     'train': DataLoader(StitchoDataset(meta_file=train_metadata, transform_fn=None, resize_dim=(512, 512)), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.workers, pin_memory=True),
        #     'train4val': DataLoader(StitchoDataset(meta_file=train_metadata, transform_fn=None, resize_dim=(512, 512)), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.workers, pin_memory=True),
        #     'test': DataLoader(StitchoDataset(meta_file=test_metadata, transform_fn=None, resize_dim=(512, 512)), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.workers, pin_memory=True),
        # }

        mask_loader = DataLoader(dataset=self.mask_set, batch_size=1)
        self.masks = [x.to(self.config.DEVICE) for x in mask_loader]

        self.results_path = os.path.join(config.PATH, 'results')

        if hasattr(config, 'RESULTS'):
            self.results_path = os.path.join(config.RESULTS)

        if hasattr(config, 'DEBUG') and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        self.inpaint_model.load()

    def save(self):
        self.inpaint_model.save()

    def inpaint(self, num_samples=1):
        loader = self.dataset['test']
        self.inpaint_model.eval()

        for index, items in enumerate(loader):

            images, masks, label, clsname = items
            images, masks = self.cuda(images, masks)

            # inpaint model
            for i in range(num_samples):
                output = self.inpaint_model(images, masks)
                output = self.postprocess(output)
                output = output.cpu()
                images = self.postprocess(images)
                images = images.cpu()
                masks = self.postprocess(masks)
                masks = masks.cpu()
                self.save_image(images, index, i, 'input')
                self.save_image(output, index, i, 'output')
                self.save_image(masks, index, i, 'mask')

    def save_image(self, output, index, i, path):
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        os.makedirs(os.path.join(self.results_path, path), exist_ok=True)
        output_paths = [os.path.join(self.results_path, path, str(index) + '_channel' + str(c) + '.png') for c in range(self.config.INPUT_CHANNELS)]
        # save tensor as n images
        for i, output_path in enumerate(output_paths):
            output_i = output[:, :, :, i]
            plt.imshow(output_i[0], cmap='gray')
            plt.axis('off')
            plt.savefig(output_path)

    def train(self):
        train_loader = self.dataset['train']

        keep_training = True
        min_error = 1
        max_epoch = int(float(self.config.MAX_EPOCHS))
        total = len(train_loader.dataset)


        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
        
        # Initialize the GradScaler for mixed precision
        scaler = torch.amp.GradScaler('cuda')

        # print(self.config.STAGE)
        # exit()

        while keep_training:
            if 1 in self.config.STAGE:
                self.inpaint_model.epoch += 1
                self.epoch = self.inpaint_model.epoch
                print('\n\nTraining proposal epoch: %d' % self.epoch)
                progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])
                # print(len(train_loader))
                # exit()
                for items in train_loader:
                    self.inpaint_model.train()

                    images, masks, label, _ = items
                    images, masks = self.cuda(images, masks)
                    # images = self.cuda(images)

                    # inpaint model
                    # train
                    with torch.amp.autocast('cuda'):
                        outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, masks)

                    # backward
                    # self.inpaint_model.backward(gen_loss, dis_loss, scaler)
                    self.inpaint_model.backward(gen_loss, dis_loss)

                    logs["epoch"] = self.epoch
                    logs["iter"] = self.inpaint_model.iteration

                    progbar.add(len(images), values=logs.items() if self.config.VERBOSE else [x for x in logs.items() if not x[0].startswith('l_')])


            # log model at checkpoints
            if self.config.LOG_INTERVAL and self.epoch % self.config.LOG_INTERVAL == 0:
                error1 = self.eval()
                aucs = self.test()

                self.log(self.epoch, aucs, error1)

                if error1 <= min_error:
                    min_error = error1
                    self.save()

            # update learning rate
            self.inpaint_model.step_learning_rate()

            if self.epoch >= max_epoch:
                break

        print('\nEnd training....')

    def eval(self):
        print('\n\nval:')
        val_loader = self.dataset['train4val']

        total = len(val_loader.dataset)
        self.inpaint_model.eval()

        mean_error = 0
        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        for index, items in enumerate(val_loader):
            images, masks = self.cuda(*items[0:2])

            # inpaint model
            error1_list, mix_out_list_x, mix_out_list_y = self.get_error_map_for_some_scales(images, self.masks, metric='L1',
                                                                           scales=self.config.SCALES, output=True)
            error1 = self.get_max_select_error(error1_list)
            mean_error += torch.mean(error1) * len(items[0]) / total

            progbar.add(len(images), values=[('index', index)])
        return mean_error.item()

    def test(self):
        print('\n\ntest:')
        self.inpaint_model.eval()

        test_loader = self.dataset['test']
        class_count = self.dataset_info['test'].get_class_count()
        class_index = {clsname: i for i, clsname in enumerate(class_count.keys())}

        total = len(test_loader.dataset)

        an_scores1 = torch.zeros(size=(total,), dtype=torch.float32, device=self.config.DEVICE)
        gt_labels = torch.zeros(size=(total,), dtype=torch.long, device=self.config.DEVICE)
        select_scale = torch.zeros(size=(total,), dtype=torch.long, device=self.config.DEVICE)
        classes = torch.zeros(size=(total,), dtype=torch.int, device=self.config.DEVICE)

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        for index, items in enumerate(test_loader):
            start_index = index * test_loader.batch_size
            end_index = start_index + items[0].shape[0]
            images, masks, label, clsname = items
            images, masks = self.cuda(images, masks)

            # print(clsname)
            # exit()

            # inpaint model
            error1_list, mix_out_list_x, mix_out_list_y = self.get_error_map_for_some_scales(images, self.masks, metric='MSE',
                                                                           scales=self.config.SCALES, output=True)
            error1, max_scale_ind = self.get_max_select_error(error1_list, need_arg=True)
            
            select_scale[start_index: end_index] = max_scale_ind
            an_scores1[start_index: end_index] = torch.mean(error1, [1, 2])
            gt_labels[start_index: end_index] = label
            classes[start_index: end_index] = torch.as_tensor(list(map(lambda x: class_index[x], clsname)))
            progbar.add(len(images), values=[('index', index)])

        aucs = {}

        an_scores1 = {clsname: an_scores1[classes == class_index[clsname]] for clsname in class_count.keys()}
        gt_labels = {clsname: gt_labels[classes == class_index[clsname]] for clsname in class_count.keys()}
        select_scale = {clsname: select_scale[classes == class_index[clsname]] for clsname in class_count.keys()}

        for class_name in class_count.keys():
            an_scores1[class_name] = (an_scores1[class_name] - torch.min(an_scores1[class_name])) / (torch.max(an_scores1[class_name]) - torch.min(an_scores1[class_name]))
            # metrics
            auc1 = evaluate(gt_labels[class_name], an_scores1[class_name])
            print(f'{class_name}: {auc1}')
            aucs[class_name] = auc1
        return aucs

    def get_error_map_coarse(self, images, mask_loader, metric='MSE'):
        if metric == 'MSE':
            error_metric = nn.MSELoss(reduction='none')
        else:
            error_metric = nn.L1Loss(reduction='none')
        with torch.no_grad():
            error = []
            raw_output = []
            sum_mask = []
            error_map_list = []
            for i, masks in enumerate(mask_loader):
                masks = masks.to(self.config.DEVICE)
                outputs = self.inpaint_model(images, masks)
                error.append(torch.mean(error_metric(outputs, images) * masks, 1))  # mean RGB channel
                raw_output.append(outputs * masks)
                sum_mask.append(masks)
            error = torch.stack(tuple(error))
            raw_output = torch.stack(tuple(raw_output))
            sum_mask = torch.stack(tuple(sum_mask))
            for index in range(0, self.config.SCALES*4, 4):
                sum_mask_t = sum_mask[index: index+4]
                sum_mask_t = torch.sum(sum_mask_t, 0)
                sum_mask_t = torch.reciprocal(sum_mask_t)
                sum_mask_t[sum_mask_t == float('inf')] = 0
                error_map_list.append(torch.sum(error[index: index+4] * sum_mask_t, 0))

            for i, error_map in enumerate(error_map_list):
                error_map_list[i] = error_map / self.scale_norm[i]
            error_map_merge = torch.stack(error_map_list)
            error_map_merge = torch.mean(error_map_merge, 0)
            # error_map_merge, _ = torch.max(error_map_merge, 0)
            mix_output = torch.sum(raw_output[0:4], 0)
        return error_map_merge, mix_output

    def get_error_map_for_some_scales(self, images, mask_loader, metric='MSE', scales=None, output=False):
        if metric == 'MSE':
            error_metric = nn.MSELoss(reduction='none')
        else:
            error_metric = nn.L1Loss(reduction='none')
        with torch.no_grad():
            error_map_list = []
            mix_output_x = []
            mix_output_y = []
            for scale in scales:
                error = []
                raw_output = []
                for masks in mask_loader[scale*4:(scale+1)*4]:
                    outputs = self.inpaint_model(images, masks)
                    if output:
                        raw_output.append(outputs*masks)
                    error.append(torch.mean(error_metric(outputs, images) * masks, 1))  # mean RGB channel
                if output:
                    raw_output = torch.stack(raw_output)
                    mix_output_x.append(torch.sum(raw_output[0:2], 0))
                    mix_output_y.append(torch.sum(raw_output[2:4], 0))
                error = torch.stack(tuple(error))
                # error_map_list.append(torch.sum(error * 0.5, 0))
                error_map_list.append(torch.max(error, 0)[0])

        return error_map_list, mix_output_x, mix_output_y


    def get_mean_merged_error(self, error_map_list):
        error_map_merge = torch.stack(error_map_list)
        error_map_merge = torch.mean(error_map_merge, 0)
        return error_map_merge

    def get_max_select_error(self, error_map_list, need_arg=False):
        error_map_list = torch.stack(error_map_list)
        mean_errors = torch.mean(error_map_list, (2, 3))
        for i, scale in enumerate(self.config.SCALES):
            # mean_errors[i] = mean_errors[i] / self.scale_norm[i]
            mean_errors[i] = mean_errors[i] - self.scale_norm[i]
        mean_errors, max_scale_ind = torch.max(mean_errors, dim=0)  # [batch]
        error_map_merge = torch.stack([error_map_list[j, i] for i, j in enumerate(max_scale_ind)])
        # error_map_merge = torch.mean(error_map_merge, 0)
        if need_arg:
            return error_map_merge, max_scale_ind
        else:
            return error_map_merge


    def get_error_map_coarse_center(self, images, mask_loader, metric='MSE'):
        if metric == 'MSE':
            error_metric = nn.MSELoss(reduction='none')
        else:
            error_metric = nn.L1Loss(reduction='none')
        error = 0
        raw_output = 0
        with torch.no_grad():
            for i, masks in enumerate(mask_loader):
                masks = masks.to(self.config.DEVICE)
                outputs = self.inpaint_model(images, masks)
                error = torch.mean(error_metric(outputs, images) * masks, 1)  # mean RGB channel
                raw_output = outputs * masks + images * (1-masks)

        return error, raw_output


    def update_norm(self):
        print('\n\nupdate normalization parameter:')
        val_loader = self.dataset['train4val']

        total = len(val_loader.dataset)
        self.inpaint_model.eval()

        mean_error_scales = {str(i): 0 for i in self.config.SCALES}
        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        for index, items in enumerate(val_loader):
            images, masks = self.cuda(*items[0:2])
            # inpaint model
            error1_list, mix_out_list_x, mix_out_list_y = self.get_error_map_for_some_scales(images, self.masks, metric='MSE',
                                                                           scales=self.config.SCALES, output=True)
            for i, scale in enumerate(self.config.SCALES):
                mean_error_scales[str(scale)] += torch.mean(error1_list[i]) * len(items[0]) / total
            progbar.add(len(images), values=[('index', index)])
        # update
        for i, scale in enumerate(self.config.SCALES):
            self.scale_norm[i] = mean_error_scales[str(scale)].item()
        print('updated norm:', self.scale_norm)

    def log(self, epoch: int, aurocs: float, error: float):
        with open(self.log_file, 'a') as f:
            f.write(f'Epoch {epoch}:\n')
            for classname, auc in aurocs.items():
                f.write(f'{classname}: AUROC = {auc}\n')
                f.write(f'{" " * len(classname)}: Error {error}\n')

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

