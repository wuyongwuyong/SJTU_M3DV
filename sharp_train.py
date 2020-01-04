from mylib.models import densesharp, metrics, losses
from mylib.utils.misc import rotation, reflection, crop, random_center, _triple

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPool3D
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


#助教参考代码中的transform类，稍稍改动输出数据维度
class Transform:
    '''The online data augmentation, including:
    1) random move the center by `move`
    2) rotation 90 degrees increments
    3) reflection in any axis
    '''

    def __init__(self, size, move):
        self.size = _triple(size)
        self.move = move

    def __call__(self, arr, aux=None):
        shape = arr.shape
        if self.move is not None:
            center = random_center(shape, self.move)
            arr_ret = crop(arr, center, self.size)
            angle = np.random.randint(4, size=3)
            arr_ret = rotation(arr_ret, angle=angle)
            axis = np.random.randint(4) - 1
            arr_ret = reflection(arr_ret, axis=axis)
            arr_ret = np.expand_dims(arr_ret, axis=0)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = rotation(aux_ret, angle=angle)
                aux_ret = reflection(aux_ret, axis=axis)
                aux_ret = np.expand_dims(aux_ret, axis=0)
                return arr_ret, aux_ret
            return arr_ret
        else:
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            arr_ret = np.expand_dims(arr_ret, axis=0)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = np.expand_dims(aux_ret, axis=0)
                return arr_ret, aux_ret
            return arr_ret

#读取训练数据
voxel_train = []                                    #设置测试数据的voxel_train
seg_train = []                                      #设置测试数据的seg_train
for i in tqdm(range(584), desc='reading'):          #展示写入训练数据的进度
    try:
        data = np.load('data/train_val/candidate{}.npz'.format(i))      #依次读取训练数据中的candidate{i}文件
    except FileNotFoundError:                                           #无该文件时直接进入下一次循环
        continue
    try:
        voxel_train = np.append(voxel_train, np.expand_dims(data['voxel'], axis=0), axis=0) #向voxel_train中添加读取的voxel向量，但是初次读取会出错
        seg_train = np.append(seg_train, np.expand_dims(data['seg'], axis=0), axis=0)       #向seg_train中添加读取的seg向量，同样初次读取时会出错
    except ValueError:
        voxel_train = np.expand_dims(data['voxel'], axis=0)
        seg_train = np.expand_dims(data['seg'], axis=0)

training_batch_size = voxel_train.shape[0]#465

#对训练数据进行数据增强扩充为原来3倍
voxel_train_new=[]
voxel_train_new = np.expand_dims(crop(voxel_train[0],(50,50,50),(32,32,32)),axis=0)
for i in range(voxel_train.shape[0]-1):
    voxel_train_new = np.append(voxel_train_new,np.expand_dims(crop(voxel_train[i+1],(50,50,50),(32,32,32)),axis=0),axis=0)
    

seg_train_new=[]
seg_train_new = np.expand_dims(crop(seg_train[0],(50,50,50),(32,32,32)),axis=0)
for i in range(seg_train.shape[0]-1):
    seg_train_new = np.append(seg_train_new,np.expand_dims(crop(seg_train[i+1],(50,50,50),(32,32,32)),axis=0),axis=0)

for i in tqdm(range(training_batch_size),desc='transforming'):
    tmp_voxel, tmp_seg = Transform(32,4)(voxel_train[i],seg_train[i])
    voxel_train_new=np.append(voxel_train_new,tmp_voxel,axis=0)
    seg_train_new=np.append(seg_train_new,tmp_seg,axis=0)
print(voxel_train_new.shape) 
print(seg_train_new.shape) 

for i in tqdm(range(training_batch_size),desc='transforming'):
    tmp_voxel, tmp_seg = Transform(32,5)(voxel_train[i],seg_train[i])
    voxel_train_new=np.append(voxel_train_new,tmp_voxel,axis=0)
    seg_train_new=np.append(seg_train_new,tmp_seg,axis=0)
print(voxel_train_new.shape) 
print(seg_train_new.shape) 



voxel_train_new = voxel_train_new.reshape(voxel_train_new.shape[0], 32, 32, 32, 1) 
seg_train_new = seg_train_new.reshape(seg_train_new.shape[0], 32, 32, 32, 1) 

print(voxel_train_new.shape) 
print(seg_train_new.shape) 


del voxel_train
del seg_train

#训练标签跟着扩充为3倍
train_label1 = pd.read_csv('data/train_val.csv').values[:, 1].astype(int)
print(train_label1.shape)


train_label = np.concatenate((train_label1,train_label1),axis=0)
train_label = np.concatenate((train_label,train_label1),axis=0)
print(train_label.shape)
train_label = to_categorical(train_label, 2)
print(train_label.shape)



a=np.random.random()#生成一个随机数
b=int(100*a)#生成一个随机数种子用于数据混洗
np.random.seed(b)
np.random.shuffle(voxel_train_new)
np.random.seed(b)
np.random.shuffle(seg_train_new)
np.random.seed(b)
np.random.shuffle(train_label)

#取前1145个做训练集，后250作为验证集
x_train,x_val = voxel_train_new[:1145],voxel_train_new[1145:]
train_seg,val_seg = seg_train_new[:1145],seg_train_new[1145:]
y_train1,y_val1 = train_label[:1145],train_label[1145:]


print(x_train.shape)
print(y_train1.shape)
print(train_seg.shape)

y_train = {"clf": y_train1, "seg": train_seg}

y_val={"clf": y_val1, "seg": val_seg}

#读取测试数据
voxel_test = []     #设置测试数据的voxel_test
seg_test = []       #设置测试数据的seg_test

for i in tqdm(range(584), desc='reading test_data'):    #展示写入测试数据的进度
    try:
        tmp = np.load('data/test/candidate{}.npz'.format(i))    #依次读取测试数据中的candidate{i}文件
    except FileNotFoundError:                                   #无该文件时直接进入下一次循环
        continue
    try:
        voxel_test = np.append(voxel_test, np.expand_dims(tmp['voxel'], axis=0), axis=0)    #向voxel_test中添加读取的voxel向量，但是初次读取会出错
        seg_test = np.append(seg_test, np.expand_dims(tmp['seg'], axis=0), axis=0)          #向seg_test中添加读取的seg向量，同样初次读取时会出错
    except ValueError:
        voxel_test = np.expand_dims(tmp['voxel'], axis=0)   #向空矩阵中写入初次读取的voxel  量
        seg_test = np.expand_dims(tmp['seg'], axis=0)   #向空矩阵中写入初次读取的seg量

seg_test = seg_test.astype(int)         #将seg布尔array转换为1/0整数
X_test= voxel_test          


X_test=X_test.astype(np.float32)


training_test_size = X_test.shape[0]  #测试数据集的数量
X_test_new=crop(X_test[0],(50,50,50),(32,32,32))

X_test_new=np.expand_dims(X_test_new,axis=0)
print(X_test_new.shape) 
test_batch_size = X_test.shape[0]
for i in tqdm(range(test_batch_size-1),desc='croping'):
    X_test_new=np.append(X_test_new,np.expand_dims(crop(X_test[i+1],(50,50,50),(32,32,32)),axis=0),axis=0)
print(X_test_new.shape)   
del X_test
X_test_new = X_test_new.reshape(X_test_new.shape[0], 32, 32, 32, 1)     #将测试数据集整合成5d张量
print(X_test_new.shape)

model = densesharp.get_compiled(output_size=2,
                                optimizer=Adam(lr=1.e-4),
                                loss={"clf": 'categorical_crossentropy',
                                      "seg": losses.DiceLoss()},
                                metrics={'clf': ['accuracy', metrics.precision, metrics.recall, metrics.fmeasure,
                                                 metrics.invasion_acc, metrics.invasion_fmeasure,
                                                 metrics.invasion_precision, metrics.invasion_recall,
                                                 metrics.ia_acc, metrics.ia_fmeasure,
                                                 metrics.ia_precision, metrics.ia_recall],
                                         'seg': [metrics.precision, metrics.recall, metrics.fmeasure]},
                                loss_weights={"clf": 1., "seg": 0.2},
                                weight_decay=0)

filepath="best_weight_sharp.h5"    
save_folder='test'


checkpointer = ModelCheckpoint(filepath='tmp/%s/weights.{epoch:02d}.h5' % save_folder, verbose=1,
                               period=1, save_weights_only=True)
#保存最佳模型
best_keeper = ModelCheckpoint(filepath='tmp/%s/best.h5' % save_folder, verbose=1, save_weights_only=False,
                              monitor='val_clf_acc', save_best_only=True, period=1, mode='max')
#进行earlystop
early_stopping = EarlyStopping(monitor='val_clf_acc', min_delta=0, mode='max',patience=30, verbose=1)

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.334, patience=10,
                               verbose=1, mode='min', epsilon=1.e-5, cooldown=2, min_lr=0)
#调整学习率
callbacks_list=[early_stopping, lr_reducer, checkpointer,best_keeper]

#训练模型
model.fit(x_train,y_train,
          shuffle=True,
          epochs=12,batch_size=64,
          callbacks=callbacks_list,
          validation_data=(x_val,y_val))
