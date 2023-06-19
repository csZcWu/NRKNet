# train_config
test_time = False

train = {}
train['train_dataset_name'] = 'LFDOF'
train['batch_size'] = 4
train['val_batch_size'] = 4
train['test_batch_size'] = 1

train['num_pre-train_epochs'] = 5000
train['num_epochs'] = 5000
train['log_epoch'] = 1
train['optimizer'] = 'Adam'
train['learning_rate'] = 1e-4

# -- for SGD -- #
train['momentum'] = 0.9
train['nesterov'] = True

# config for save , log and resume
train['sub_dir'] = '/NRKNet_' + train['train_dataset_name']
train['resume'] = './save/NRKNet_' + train['train_dataset_name'] + '/0'
train['resume_epoch'] = None  # None means the last epoch
train['resume_optimizer'] = './save/NRKNet_' + train['train_dataset_name'] + '/0'
data_offset = 'G:/datasets/'

net = {}
net['xavier_init_all'] = True
net['num_res'] = 1
net['num_kernels'] = 20
net['kernel_mode'] = 'FG'
net['in_ch'] = 3

loss = {}
loss['weight_l2_reg'] = 0

test = {}
test['dataset'] = 'RTF'  # DPDD, LFDOF, RTF, RealDOF
