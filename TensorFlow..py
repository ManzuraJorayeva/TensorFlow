#!/usr/bin/env python
# coding: utf-8

# In[12]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[13]:


print(tf.__version__)


# In[14]:


hello = tf.constant("Hello")


# In[15]:


world = tf.constant("World")


# In[16]:


type(hello)


# In[17]:


print(hello)


# In[18]:


with tf.Session() as sess:
    
    result = sess.run(hello+world)


# In[20]:


print(result)


# In[21]:


a = tf.constant(10)


# In[22]:


b = tf.constant(20)


# In[23]:


type(a)


# In[24]:


a +b 


# In[25]:


a +b 


# In[26]:


a +b 


# In[27]:


with tf.Session() as sess:
    
    result = sess.run(a+b)


# In[28]:


result


# In[29]:


const = tf.constant(10)


# In[30]:


fill_mat = tf.fill((4,4),(10))


# In[31]:


myzeros = tf.zeros((4,4))


# In[32]:


myones = tf.ones((4,4))


# In[33]:


myrandn = tf.random_normal((4,4), mean=0,stddev=1.0)


# In[34]:


myrandu = tf.random_uniform((4,4), minval=0, maxval=1)


# In[ ]:





# In[35]:


my_ops = [const, fill_mat, myzeros, myones, myrandn, myrandu]


# In[36]:


sess = tf.InteractiveSession()


# In[38]:


for op in my_ops:
    print(sess.run(op))
    print('\n')


# In[39]:


a = tf.constant([[1,2] ,                    
                 [3,4]])


# In[40]:


a.get_shape()


# In[42]:


b = tf.constant([[10],[100]])


# In[43]:


b.get_shape()


# In[44]:


result = tf.matmul(a,b)


# In[45]:


sess.run(result)


# In[46]:


result.eval()


# In[ ]:


#TF Graphs


# In[89]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[57]:


n1 = tf.constant(1)


# In[58]:


n2 = tf.constant(2)


# In[59]:


n3 = n1 + n2


# In[61]:


with tf.Session() as sess:
    result = sess.run(n3) 


# In[62]:


print(result)


# In[63]:


print(n3)


# In[64]:


print(tf.get_default_graph())


# In[65]:


g = tf.Graph()


# In[66]:


print(g)


# In[67]:


graph_one = tf.get_default_graph()


# In[68]:


print(graph_one)


# In[69]:


graph_two = tf.Graph()


# In[70]:


print(graph_two)


# In[71]:


with graph_two.as_default():
    print(graph_two is tf.get_default_graph()) 


# In[72]:


print(graph_two is tf.get_default_graph()) 


# In[73]:


#Variables and Placeholders


# In[87]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[75]:


sess = tf.InteractiveSession()


# In[77]:


my_tensor = tf.random_uniform((4,4),0,1)


# In[78]:


my_tensor


# In[79]:


my_var = tf.Variable(initial_value=my_tensor)


# In[80]:


print(my_var)


# In[88]:


init = tf.global_variables_initializer()


# In[90]:


sess.run(init)


# In[91]:


sess.run(my_var)


# In[92]:


ph = tf.placeholder(tf.float32, shape=(None,5))


# In[93]:


#Tensorflow A Neural Network Pat One


# In[156]:


import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[98]:


np.random.seed(101)
tf.set_random_seed(101)


# In[101]:


rand_a = np.random.uniform(0,100,(5,5))
rand_a


# In[102]:


rand_b = np.random.uniform(0,100,(5,1))
rand_b


# In[103]:


a = tf.placeholder(tf.float32)


# In[104]:


b = tf.placeholder(tf.float32)


# In[105]:


add_op = a+b


# In[106]:


mul_op = a * b


# In[113]:


with tf.Session() as sess:
    
    add_result = sess.run(add_op,feed_dict={a:rand_a,b:rand_b})
    print(add_result)
    print('\n')
    mult_result = sess.run(mul_op,feed_dict={a:rand_a,b:rand_b})
    print(mult_result)


# In[114]:


#Tensorflow A Neural Network Part Two


# In[115]:


n_features = 10
n_dense_neurons = 3


# In[116]:


x = tf.placeholder(tf.float32,(None,n_features))


# In[127]:


W = tf.Variable(tf.random.normal([n_features,n_dense_neurons]))

b = tf.Variable(tf.ones([n_dense_neurons]))


# In[128]:


xW = tf.matmul(x,W)


# In[129]:


z = tf.add(xW,b)


# In[130]:


a = tf.sigmoid(z)


# In[131]:


init =tf.global_variables_initializer()


# In[133]:


with tf.Session() as sess:
    
    sess.run(init)
    
    layer_out = sess.run(a,feed_dict={x:np.random.random([1,n_features])})


# In[134]:


print(layer_out)


# In[135]:


#Simple Regression Example


# In[136]:


x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)


# In[137]:


x_data


# In[138]:


y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)


# In[139]:


y_label


# In[140]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[141]:


plt.plot(x_data,y_label,'*')


# In[142]:


np.random.rand(2)


# In[143]:


m = tf.Variable(0.84)
b = tf.Variable(0.38)


# In[145]:


error = 0

for x,y in zip(x_data,y_label):
    
    y_hat = m*x +b
    
    error += (y-y_hat)**2


# In[146]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)


# In[147]:


init = tf.global_variables_initializer()


# In[151]:


with tf.Session() as sess:
    
    sess.run(init)
    
    training_steps = 100
    
    for i in range(training_steps):
        
        sess.run(train)
        
    final_slope, final_intercept = sess.run([m,b])


# In[152]:


x_test = np.linspace(-1,11,10)

#y= mx+b
y_pred_plot = final_slope*x_test + final_intercept

plt.plot(x_test,y_pred_plot,'r')
plt.plot(x_data,y_label,'*')


# In[153]:


#Tensorflow Regression Example Part One


# In[154]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[158]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[159]:


x_data = np.linspace(0.0,10.0,1000000)


# In[160]:


noise = np.random.randn(len(x_data))


# In[161]:


noise


# In[163]:


#y=mx+b
#b=5


# In[164]:


y_true = (0.5 * x_data) + 5 + noise


# In[165]:


x_df = pd.DataFrame(data=x_data,columns=['X Data'])


# In[168]:


y_df = pd.DataFrame(data=y_true,columns=['Y'])


# In[170]:


y_df.head()


# In[171]:


my_data = pd.concat([x_df,y_df],axis=1)


# In[173]:


my_data.head()


# In[175]:


my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')


# In[176]:


batch_size = 8


# In[178]:


np.random.randn(2)


# In[179]:


m = tf.Variable(0.46)
b = tf.Variable(0.38)


# In[180]:


xph = tf.placeholder(tf.float32,[batch_size])


# In[181]:


yph = tf.placeholder(tf.float32,[batch_size])


# In[182]:


y_model = m*xph + b


# In[183]:


error = tf.reduce_sum(tf.square(yph-y_model))


# In[185]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)


# In[186]:


init = tf.global_variables_initializer()


# In[188]:


with tf.Session() as sess:
    
    sess.run(init)
    
    batches = 1000
    
    for i in range(batches):
        
        rand_ind = np.random.randint(len(x_data),size=batch_size)
        
        feed = {xph:x_data[rand_ind], yph:y_true[rand_ind]}
        
        sess.run(train,feed_dict=feed)
        
    model_m, model_b = sess.run([m,b])


# In[189]:


model_m


# In[190]:


model_b


# In[191]:


y_hat = x_data*model_m + model_b


# In[192]:


my_data.sample(250).plot(kind='scatter',x='X Data',y='Y')
plt.plot(x_data,y_hat,'r')


# In[193]:


#TensorFlow Regression Example Part Two


# In[194]:


#TF ESTIMATOR


# In[195]:


feat_cols = [tf.feature_column.numeric_column('x',shape = [1])]


# In[196]:


estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)


# In[197]:


from sklearn.model_selection import train_test_split


# In[198]:


x_train, x_eval, y_train, y_eval=train_test_split(x_data,y_true,test_size=0.3,random_state=101)


# In[199]:


print(x_train.shape)


# In[200]:


x_eval.shape


# In[201]:


input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8,num_epochs=None,shuffle=True)


# In[202]:


train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8,num_epochs=1000,shuffle=False)


# In[203]:


eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval},y_eval,batch_size=8,num_epochs=1000,shuffle=False)


# In[204]:


estimator.train(input_fn=input_func,steps=1000)


# In[205]:


train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)


# In[207]:


eval_metrics = estimator.evaluate(input_fn=eval_input_func,steps=1000)


# In[208]:


print('TRAININ DATA METRICS')
print(train_metrics)


# In[209]:


print('EVAL METRICS')
print(eval_metrics)


# In[210]:


brand_new_data = np.linspace(0,10,10)


# In[211]:


input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data},shuffle=False)


# In[213]:


list(estimator.predict(input_fn=input_fn_predict))


# In[215]:


predictions = []

for pred in estimator.predict(input_fn=input_fn_predict):
    
    predictions.append(pred['predictions'])


# In[216]:


predictions


# In[219]:


my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')
plt.plot(brand_new_data,predictions,'r*')


# In[220]:


#Tensorflow Classification Example Part One


# In[262]:


import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[264]:


diabetes = pd.read_csv('pima-indians-diabetes.csv')


# In[265]:


diabetes.head()


# In[267]:


diabetes.columns


# In[268]:


cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']


# In[270]:


diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x:(x-x.min()) / (x.max()-x.min() ))


# In[271]:


diabetes.head()


# In[293]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[284]:


diabetes.columns


# In[285]:


num_preg  = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')


# In[286]:


assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])


# In[277]:


#assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)


# In[287]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[289]:


diabetes['Age'].hist(bins=20)


# In[295]:


age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])


# In[296]:


feat_cols = [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree,assigned_group,age_bucket]


# In[297]:


#TRAIN TEST SPLIT


# In[301]:


x_data = diabetes.drop('Class',axis=1)


# In[302]:


x_data


# In[303]:


x_data.head()


# In[304]:


labels = diabetes['Class']


# In[305]:


labels


# In[310]:


from sklearn.model_selection import train_test_split


# In[314]:


X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.33, random_state=101)


# In[315]:


#labels


# In[320]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)


# In[321]:


model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)


# In[322]:


model.train(input_fn=input_func,steps=1000)


# In[323]:


eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1,shuffle=False)


# In[324]:


results = model.evaluate(eval_input_func)


# In[325]:


results


# In[326]:


pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10,num_epochs=1,shuffle=False)


# In[327]:


predictions = model.predict(pred_input_func)


# In[328]:


my_pred = list(predictions)


# In[329]:


my_pred


# In[331]:


dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)


# In[332]:


embedded_group_col = tf.feature_column.embedding_column(assigned_group,dimension=4)


# In[333]:


feat_cols =  [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree,embedded_group_col,age_bucket]


# In[334]:


input_func = tf.estimator.inputs.pandas_input_fn(X_train,y_train,batch_size=10,num_epochs=1,shuffle=True)


# In[343]:


dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,20,20,10],feature_columns=feat_cols,n_classes=2)


# In[336]:


dnn_model.train(input_fn=input_func,steps=1000)


# In[340]:


eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)


# In[342]:


dnn_model.evaluate(eval_input_func)


# In[344]:


#TF REGRESSION EXERCISE


# In[345]:


import pandas as pd


# In[347]:


housing = pd.read_csv('cal_housing_clean.csv') 


# In[348]:


housing.head()


# In[349]:


#


# In[350]:


#


# In[352]:


y_val = housing['medianHouseValue']


# In[354]:


x_data = housing.drop('medianHouseValue', axis=1)


# In[356]:


from sklearn.model_selection import train_test_split


# In[361]:


X_train, X_test, y_train, y_test = train_test_split(x_data,y_val,test_size=0.3,random_state=101)


# In[363]:


from sklearn.preprocessing import MinMaxScaler


# In[364]:


scaler = MinMaxScaler()


# In[365]:


scaler.fit(X_train)


# In[366]:


X_train = pd.DataFrame(data=scaler.transform(X_train),columns=X_train.columns,index=X_train.index)


# In[367]:


X_test = pd.DataFrame(data=scaler.transform(X_test),columns=X_test.columns,index=X_test.index)


# In[368]:


#Feature Columns


# In[369]:


housing.columns


# In[370]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[371]:


age = tf.feature_column.numeric_column('housingMedianAge')


# In[372]:


rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')


# In[374]:


feat_cols = [age,rooms,bedrooms,pop,households,income]


# In[375]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)


# In[376]:


model = tf.estimator.DNNRegressor(hidden_units=[6,6,6],feature_columns=feat_cols)


# In[398]:


model.train(input_fn=input_func,steps=1000)


# In[399]:


predict_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10,num_epochs=1,shuffle=False)


# In[400]:


pred_gen = model.predict(predict_input_func)


# In[401]:


predictions = list(pred_gen)


# In[408]:


predictions


# In[409]:


#TF CLASSIFICATION


# In[410]:


import pandas as pd


# In[411]:


census = pd.read_csv('census_data.csv')


# In[412]:


census.head()


# In[413]:


census['income_bracket'].unique()


# In[415]:


def label_fix(label):
    
    if label ==' <=50K':
        return 0
    else:
        return 1


# In[416]:


census['income_bracket'] = census['income_bracket'].apply(label_fix)


# In[417]:


census.head()


# In[418]:


from sklearn.model_selection import train_test_split


# In[420]:


x_data = census.drop('income_bracket',axis=1)


# In[421]:


y_labels =census['income_bracket']


# In[422]:


X_train, X_test, y_train, y_test = train_test_split(x_data,y_labels,test_size=0.3,random_state=101)


# In[423]:


census.columns


# In[431]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[432]:


gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)


# In[433]:


age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")


# In[434]:


feat_cols = [gender,occupation,marital_status,relationship,education,workclass,native_country,
            age,education_num,capital_gain,capital_loss,hours_per_week]


# In[435]:


input_func=tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100,num_epochs=None,shuffle=True)


# In[436]:


model = tf.estimator.LinearClassifier(feature_columns=feat_cols)


# In[437]:


#Train Data


# In[438]:


model.train(input_fn=input_func,steps=1000)


# In[439]:


pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)


# In[440]:


pred_gen = model.predict(input_fn=pred_fn)


# In[441]:


predictions = list(pred_gen)


# In[444]:


predictions[0]


# In[446]:


final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])


# In[448]:


final_preds


# In[449]:


from sklearn.metrics import classification_report


# In[451]:


print(classification_report(y_test,final_preds))


# In[ ]:




