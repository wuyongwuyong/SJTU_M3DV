from mylib.models import densesharp, metrics, losses,densenet
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
#用于mixup的数据生成器
class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)
        X = np.expand_dims(X,axis=-1)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y

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



#voxel_train=voxel_train.astype(float)/255
training_batch_size = voxel_train.shape[0]#465

voxel_train=voxel_train*seg_train #抠出结节

voxel_train_new=[]
voxel_train_new = np.expand_dims(crop(voxel_train[0],(50,50,50),(32,32,32)),axis=0)
for i in range(voxel_train.shape[0]-1):
    voxel_train_new = np.append(voxel_train_new,np.expand_dims(crop(voxel_train[i+1],(50,50,50),(32,32,32)),axis=0),axis=0)
    
train_label1 = pd.read_csv('data/train_val.csv').values[:, 1].astype(int)
    

    
X_train_new,x_val,train_label, y_val1 =train_test_split(voxel_train_new,train_label,test_size=100,shuffle=True,stratify=train_label)

x_val = x_val.reshape(x_val.shape[0], 32, 32, 32, 1) 
print(X_train_new.shape)
print(train_label.shape)

#数据增强
for i in tqdm(range(training_batch_size),desc='transforming'):
    tmp_voxel = Transform(32,4)(voxel_train[i])
    voxel_train_new=np.append(voxel_train_new,tmp_voxel,axis=0)
    seg_train_new=np.append(seg_train_new,tmp_seg,axis=0)
print(voxel_train_new.shape) 
print(seg_train_new.shape) 

for i in tqdm(range(training_batch_size),desc='transforming'):
    tmp_voxel= Transform(32,4)(voxel_train[i])
    voxel_train_new=np.append(voxel_train_new,tmp_voxel,axis=0)
print(voxel_train_new.shape) 
print(seg_train_new.shape) 


del voxel_train
del seg_train

print(train_label1.shape)

#训练标签也扩充
train_label = np.concatenate((train_label1,train_label1),axis=0)
train_label = np.concatenate((train_label,train_label1),axis=0)
print(train_label.shape)
train_label = to_categorical(train_label, 2)
print(train_label.shape)
#将训练集划分

a=np.random.random()
b=int(100*a)
x_train,x_val,y_train1, y_val1 =train_test_split(voxel_train_new,train_label,random_state=np.random.seed(b),
                                                   test_size=250,shuffle=True,stratify=train_label)


train_seg,val_seg,y_train1, y_val1 =train_test_split(seg_train_new,train_label,random_state=np.random.seed(b),
                                                   test_size=250,shuffle=True,stratify=train_label)
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
X_test= voxel_test*seg_test          #将结节抠出来


X_test=X_test.astype(np.float32)
#X_test=X_test/128.-1.
#X_test = np.concatenate((X_test,np.transpose(X_test,(0,2,1,3))),axis=0)    #将训练集的xy转置得到新数据集以扩充数据
#print(X_test.shape)
training_test_size = X_test.shape[0]  #训练数据集的数量
X_test_new=crop(X_test[0],(50,50,50),(32,32,32))

X_test_new=np.expand_dims(X_test_new,axis=0)
print(X_test_new.shape) 
test_batch_size = X_test.shape[0]
for i in tqdm(range(test_batch_size-1),desc='croping'):
    X_test_new=np.append(X_test_new,np.expand_dims(crop(X_test[i+1],(50,50,50),(32,32,32)),axis=0),axis=0)
print(X_test_new.shape)   
del X_test
X_test_new = X_test_new.reshape(X_test_new.shape[0], 32, 32, 32, 1)     #将训练数据集整合成5d张量
print(X_test_new.shape)

model = densenet.get_compiled(optimizer=Adam(lr=1.e-4),
                              loss='categorical_crossentropy',
                              metrics=["categorical_accuracy", invasion_acc,
                                       invasion_precision, invasion_recall, invasion_fmeasure])

filepath="best_weight_net.h5"    
save_folder='test'

checkpointer = ModelCheckpoint(filepath='tmp/%s/weights.{epoch:02d}.h5' % save_folder, verbose=1,
                               period=1, save_weights_only=True)
best_keeper = ModelCheckpoint(filepath='tmp/%s/best.h5' % save_folder, verbose=1, save_weights_only=False,
                              monitor='val_clf_acc', save_best_only=True, period=1, mode='max')
early_stopping = EarlyStopping(monitor='val_clf_acc', min_delta=0, mode='max',patience=20, verbose=1)

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.442, patience=5,
                               verbose=1, mode='min', epsilon=1.e-5, cooldown=2, min_lr=0)

callbacks_list=[early_stopping, lr_reducer, checkpointer,best_keeper]

training_generator = MixupGenerator(X_train_new, train_label, batch_size=32, alpha=0.4)()
model.fit_generator(generator=training_generator,
                         steps_per_epoch=1145//32+1,shuffle=True,
                         epochs=40,
                         callbacks=callbacks_list,
                         validation_data=(x_val,y_val))