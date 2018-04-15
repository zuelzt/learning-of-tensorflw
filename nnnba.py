#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:03:29 2018

@author: Rorschach
@mail: 188581221@qq.com
"""
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import os
import numpy as np
import pandas as pd
import time

data_folder = os.path.join('/Users/zt', 'Desktop/Master File/practice of python/Data',
                           'NBA')
data_filename = os.path.join(data_folder, 'leagues_NBA_2014_games_games.csv')

# 将Date转化为日期对象，并跳过第一行空值
results = pd.read_csv(data_filename, parse_dates=['Date'], skiprows=[0,])

# 命名列
results.columns = ["Date", "Score Type", "Visitor Team", "VisitorPts", "Home Team", "HomePts", "OT?", "Notes"]

# label
results['HomeWin'] = results['VisitorPts'] < results['HomePts']
y = results['HomeWin'].values
y = np.array(y, dtype='bool')

# 设定特征
#特征1  上一场获胜
results['HomeLastWin'] = 0
results['VisitorLastWin'] = 0

from collections import defaultdict
won_last = defaultdict(int)   
for index, item in results.iterrows():  
    home_team = item['Home Team']
    visitor_team = item['Visitor Team']
    item['HomeLastWin'] = won_last[home_team]   
    item['VisitorLastWin'] = won_last[visitor_team]
    results.ix[index] = item   #写入每行
    #每次比赛后更新字典
    won_last[home_team] = int(item['HomeWin'])    
    won_last[visitor_team] = int(not item['HomeWin'])  

#特征2 win streaks 连胜
results['HomeWinStreak'] = 0
results['VisitorWinStreak'] = 0
win_streak = defaultdict(int)

for index, item in results.iterrows():
    home_team = item['Home Team']
    visitor_team = item['Visitor Team']
    item['HomeWinStreak'] = win_streak[home_team]
    item['VisitorWinStreak'] = win_streak[visitor_team]
    results.ix[index] = item
    #
    if item['HomeWin']:
        win_streak[home_team] += 1
        win_streak[visitor_team] = 0
    else:
        win_streak[home_team] = 0
        win_streak[visitor_team] += 1

#特征3 去年战绩
ladder_filename = os.path.join(data_folder, '2013.csv')
ladder = pd.read_csv(ladder_filename, skiprows=[0])

results['HomeTeamRanksHigher'] = 0

for index, item in results.iterrows():
    home_team = item['Home Team']
    visitor_team = item['Visitor Team']
    if home_team =='New Orleans Pelicans':
        home_team = 'New Orleans Hornets'
    elif visitor_team == 'New Orleans Pelicans':
        visitor_team = 'New Orleans Hornets'
    home_rank = ladder[ladder['Team'] == home_team]['Rk'].values[0]
    #不加 [0] 是一个 array ，加了是一个int
    visitor_rank = ladder[ladder['Team'] == visitor_team]['Rk'].values[0]
    item['HomeTeamRanksHigher'] = int(home_rank > visitor_rank)
    #越高越弱
    results.ix[index] = item

#特征4 上次交手
last_match_winner = defaultdict(int)
results['HomeTeamWonLast'] = 0

for index, item in results.iterrows():
    home_team = item['Home Team']
    visitor_team = item['Visitor Team']
    #不管谁主场谁客场，统一为一个key
    teams = tuple(sorted([home_team, visitor_team]))
    #sorted 后是一个[]
    item['HomeTeamWonLast'] = 1 if last_match_winner[teams] == home_team else 0
    results.ix[index] = item
    #
    winner = item['Home Team'] if item['HomeWin'] else item['Visitor Team']
    last_match_winner[teams] = winner
    
#特征5 将字符型球队名字转化为整型
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoding = LabelEncoder()
encoding.fit(results['Home Team'].values)  #转化
#组合
home_teams = encoding.transform(results['Home Team'].values)
visitor_teams = encoding.transform(results['Visitor Team'].values)
#均转化为 array([x,x,....,x])
#vstack 行合并  .T为转置
X_teams = np.vstack([home_teams, visitor_teams]).T
#避免分类器认为 1，2 相似， 1，10 不同，采用 OneHot
onehot = OneHotEncoder()
X_teams = onehot.fit_transform(X_teams).todense()  #todense() 得到matrix

# 其余特征
#比赛时间间隔,过去五场，历史交手
##时间间隔
results['Home Game Seq'] = 0
results['Visitor Game Seq'] = 0
LastGame = defaultdict(int)
base = results['Date'][0]
##过去五场
results['Home Last5'] = float(0)
results['Visitor Last5'] = float(0)
##历史交手
results['WhoWin'] = int(0)

for index, item in results.iterrows():
    home_team = item['Home Team']
    visitor_team = item['Visitor Team']
    
    ##时间间隔
    now = item['Date']
    delta = now - base
    delta = delta.total_seconds()
    home_last_game = LastGame[home_team]
    visitor_last_game = LastGame[visitor_team]
    home_delta = delta - home_last_game 
    visitor_delta = delta - visitor_last_game
    home_d = int(home_delta / (60 * 60 * 24))
    visitor_d = int(visitor_delta / (60 * 60 * 24))
    
    ##过去五场
    zjw = [0, 0, 0, 0, 0]
    zt = [0, 0, 0, 0, 0]
    ###home
    a = results[results['Home Team'] == home_team]
    b = results[results['Visitor Team'] == home_team]
    m = pd.merge(a, b, on=list(a.columns), how= 'outer', sort=False,
                 left_index=True, right_index=True)
    m.insert(0, 'No', range(m.shape[0]))
    now = int(m[m.index.values == index]['No'])
    if now > 5:
        for i in range(5):
            gamei = now - i - 1
            want = m[m['No'] == gamei]
            if (want['Home Team'] == home_team).values[0] and (want['HomeWin'] == True).values[0]:
                zjw[i] = 1
            elif (want['Visitor Team'] == home_team).values[0] and (want['HomeWin'] == False).values[0]:
                zjw[i] = 1
            else:
                zjw[i] = 0
    else:
        zjw = 0.5
    ###visitor
    c = results[results['Home Team'] == visitor_team]
    d = results[results['Visitor Team'] == visitor_team]
    n = pd.merge(c, d, on=list(c.columns), how= 'outer', sort=False,
                 left_index=True, right_index=True)  #保留index
    n.insert(0, 'No', range(n.shape[0]))
    nnow = int(n[n.index.values == index]['No'])
    if nnow > 5:
        for i in range(5):
            gamei = nnow - i - 1
            want = n[n['No'] == gamei]
            if (want['Home Team'] == visitor_team).values[0] and (want['HomeWin'] == True).values[0]:
                zt[i] = 1
            elif (want['Visitor Team'] == visitor_team).values[0] and (want['HomeWin'] == False).values[0]:
                zt[i] = 1
            else:
                zt[i] = 0
    else:
        zt = 0.5
    
    ##历史交手
    T1 = list(np.repeat(False, 1230))
    T2 = list(np.repeat(False, 1230))
    for x in range(len((results['Home Team'] == home_team).values)):
        T1[x] = (results['Home Team'] == home_team).values[x] and (results['Visitor Team'] == visitor_team).values[x]
        T2[x] = (results['Home Team'] == visitor_team).values[x] and (results['Visitor Team'] == home_team).values[x]
    a = results[T1] 
    b = results[T2]
    mix = pd.merge(a, b, on=list(a.columns), how='outer', sort=False,
                   left_index=True, right_index=True)
    mix.insert(0, 'No', range(mix.shape[0]))
    nnnow = int(mix[mix.index.values == index]['No'])
    if nnnow > 0:
        win = list(np.repeat([0.5],nnnow))
        for i in range(nnnow):
            use = mix[mix['No'] == i]
            if (use['HomeWin'] == True).values[0]:
                win[i] = 1 #home win
            else:
                win[i] = 0 #visitor win
    else:
        win = 0.5
    winwin = np.mean(win)
    if winwin >= 0.5:
        ww = 1
    else:
        ww = 0
    
    ##时间间隔
    LastGame[home_team] = delta
    LastGame[visitor_team] = delta
    
    item['Home Game Seq'] = home_d
    item['Visitor Game Seq'] = visitor_d
    
    item['Home Last5'] = np.mean(zjw)
    item['Visitor Last5'] = np.mean(zt)
    
    item['WhoWin'] = ww
    
    results.ix[index] = item

# x
x = results[['HomeLastWin', 'VisitorLastWin', 'HomeWinStreak',
             'VisitorWinStreak', 'HomeTeamRanksHigher', 'HomeTeamWonLast',
             'Home Game Seq', 'Visitor Game Seq', 'Home Last5', 'Visitor Last5',
             'WhoWin']].values

# tf

# hyperparameter
learning_rate = 0.001
train_step = 3000
display_step = 30
log_path = '/Users/zt/Desktop/logs/'

# parameter
n_input = 11
n_layer_1 = 8
n_layer_2 =4
n_output = 1

# placeholder
with tf.name_scope('Placeholder'):
    X = tf.placeholder(tf.float32, [None, n_input], name='X')
    y = tf.placeholder(tf.float32, [None, n_output], name='y')
    
# weights, biases
with tf.name_scope('Parameter'):
    with tf.name_scope('Weights'):
        w = {'w_i': tf.Variable(tf.random.randn([n_input, n_layer_1]), name='w_i'),
             'w_1': tf.Variable(tf.random.randn([n_layer_1, n_layer_2]), name='w_1'),
             'w_2': tf.Variable(tf.random.randn([n_layer_2, n_output]), name='w_2')}
        for name in w.keys():
            tf.summary.histogram(name, w[name])
    with tf.name_scope('Biases'):
        b = {'b_i': tf.Variable(tf.random.randn([n_layer_1]), name='b_i'),
             'b_1': tf.Variable(tf.random.randn([n_layer_2]), name='b_1'),
             'b_2': tf.Variable(tf.random.randn([n_output]), name='b_2')}
        for name in b.keys():
            tf.summary.histogram(name, b[name])
            
# model
def models(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(X, w['w_i']), b['b_i']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, w['w_1']), b['b_1']))
    layer3 = tf.nn.softmax(tf.add(tf.matmul(layer2, w['w_2']), b['b_2']))
    return layer3

with tf.name_scope('Model'):
    pred = models(X)   
    
# cost
with tf.name_scope('Cost'):
    cost = tf.reduce_mean(tf.pow(y-pred, 2))
    tf.summary.scalar('cost', cost)
 
# accuracy
with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(y, tf.bool), pred>0.5), tf.float32))
# optimizer
with tf.name_scope('Optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
# before
sess = tf.Session()
init = tf.global_variables_initializer()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_path, sess.graph)

# train
begin = time.time()
sess.run(init)
for step in train_step:
    _, mer, costs, acc = sess.run(optimizer, merged, cost, accuracy, feed_dict={X: x})
    if (step+1) % display_step == 0:
       writer.add_summary(mer, step)
       print('Step:{0} --- cost:{1:.4f} --- accuracy:{2:.4f}'.format(step+1, costs, acc))
end = time.time()
print('Total: {} s !'.format(end-begin))














