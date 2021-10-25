import torch
import pandas as pd
from math import log10
from CSNet.model import RFAN
from utility import progress_bar
import torch.backends.cudnn as cudnn

results = {'loss': [], 'set5': [], 'set11': [], 'set14': []}


class Trainer(object):
    def __init__(self, config, training_loader, testing_loader1, testing_loader2, testing_loader3):
        super(Trainer, self).__init__()
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.output_path = './epochs'
        self.nEpochs = config.nEpochs
        self.sampling_rate = config.samplingRate
        self.sampling_point = config.samplingPoint
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.training_loader = training_loader
        self.testing_loaderset5 = testing_loader1
        self.testing_loaderset11 = testing_loader2
        self.testing_loaderset14 = testing_loader3
        self.num_blocks = config.resBlock

    def build_model(self):
        self.model = RFAN(num_channels=1, num_features=self.sampling_point, num_blocks=self.num_blocks).to(self.device)
        self.model.weight_init(mean=0.0, std=0.02)
        torch.manual_seed(self.seed)
        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterionL1.cuda()
            self.criterionMSE.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5)

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterionMSE(self.model(data), data)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.6f' % (train_loss / (batch_num + 1)))
        return format(train_loss / len(self.training_loader))

    def testset5(self):
        self.model.eval()
        avg_psnr = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loaderset5):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                mse = self.criterionMSE(prediction, data)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                progress_bar(batch_num, len(self.testing_loaderset5), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))
        return format(avg_psnr / len(self.testing_loaderset5))

    def testset11(self):
        self.model.eval()
        avg_psnr = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loaderset11):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                mse = self.criterionMSE(prediction, data)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                progress_bar(batch_num, len(self.testing_loaderset11), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))
        return format(avg_psnr / len(self.testing_loaderset11))

    def testset14(self):
        self.model.eval()
        avg_psnr = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loaderset14):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                mse = self.criterionMSE(prediction, data)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                progress_bar(batch_num, len(self.testing_loaderset14), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))
        return format(avg_psnr / len(self.testing_loaderset14))

    def run(self):
        self.build_model()
        epoch = 0
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            loss = self.train()
            results['loss'].append(loss)
            avg_psnrset5 = self.testset5()
            avg_psnrset11 = self.testset11()
            avg_psnrset14 = self.testset14()
            results['set5'].append(avg_psnrset5)
            results['set11'].append(avg_psnrset11)
            results['set14'].append(avg_psnrset14)
            self.scheduler.step()
            model_out_path = self.output_path + "/model_path_" + str(epoch) + ".pth"
            torch.save(self.model, model_out_path)
        out_path = 'data_results/'
        data_frame = pd.DataFrame(
            # data={'Loss': results['loss'], 'set11': results['set11']},
            data={'Loss': results['loss'], 'set5': results['set5'], 'set11': results['set11'],
                  'set14': results['set14']},
            index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'CSNet_testmse' + str(self.sampling_rate) + str(self.num_blocks) + '.csv',
                          index_label='Epoch')
