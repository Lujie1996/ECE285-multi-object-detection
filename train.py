import dataset
import model
import stats
import util
import loss
import datetime
import torch
import nntools as nt
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_root_dir = '/datasets/ee285f-public/PascalVOC2012'

train_set = dataset.VOCDataset(dataset_root_dir, S=7)
val_set = dataset.VOCDataset(dataset_root_dir, mode='val', S=7)

lr = 1e-4
net = model.ResYOLO(num_classes=20)
net = net.to(device)
adam = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
#scheduler = torch.optim.lr_scheduler.LambdaLR(adam, lr_lambda, last_epoch=-1)
stats_manager = stats.ClassificationStatsManager()
exp = nt.Experiment(net, train_set, val_set, adam, stats_manager, batch_size=64,
               output_dir="Demo", perform_validation_during_training=True)

fig, axes = plt.subplots(ncols=2, figsize=(7, 3))
print(datetime.datetime.now())
exp.run(num_epochs=135, plot=lambda exp: util.plot(exp, fig=fig, axes=axes))

