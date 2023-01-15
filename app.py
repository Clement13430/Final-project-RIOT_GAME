# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 20:36:24 2023
@author: cleme
"""

import requests
import numpy as np
import time 
import tensorflow as tf
import pdb
import sys
import urllib, json
from tensorflow import keras
from tensorflow.keras import layers
import datetime
from datetime import date
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from keras import layers, initializers
import os
import gradio as gr


url_all_queue_Id = 'https://static.developer.riotgames.com/docs/lol/queues.json'
queue_Id_solo_duo = 420
url = 'http://ddragon.leagueoflegends.com/cdn/12.23.1/data/en_US/champion.json'
response = urllib.request.urlopen(url)
all_champions = json.loads(response.read())
number_of_champions = len(all_champions['data'])
champ_Id = {}
for i, champions  in enumerate(all_champions['data'].keys()):
    champ_Id[str(champions).lower()] = i
 
    
model_to_use = keras.models.load_model('default_model_single_layer')


def normalization_division(division, above_diamond):
    if above_diamond == True:
        return 0
    if division == 'I':
        mult = 3
    if division == 'II':
        mult = 2
    if division == 'III':
        mult = 1
    if division == 'IV':
        mult = 0  
        
    return mult

def identity(x):
    return x
   
def normalization_rank(info, function = identity):
    mult_rank = 0
    tier = info['tier']
    division = info['rank']
    LP = info['leaguePoints']
    if tier == 'IRON':
        mult_rank = 0
        mult_div = normalization_division(division, False)
    if tier == 'BRONZE':
        mult_rank = 1
        mult_div = normalization_division(division, False)
    if tier == 'SILVER':
        mult_rank = 2
        mult_div = normalization_division(division, False)
    if tier == 'GOLD':
        mult_rank = 3    
        mult_div = normalization_division(division, False)
    if tier == 'PLATINUM':
        mult_rank = 4
        mult_div = normalization_division(division, False)
    if tier == 'DIAMOND':
        mult_rank = 5
        mult_div = normalization_division(division, False)
    if tier == 'MASTER' or tier == 'CHALLENGER' or tier == 'GRANDMASTER' :
        mult_rank = 6
        mult_div = normalization_division(division, True)
    rank = mult_rank * 400 + mult_div * 100 + LP
    rank = function(rank)/ function(4200)

    return rank


def define_player(tier, division, Leaguepoints, champion_name):
    info = {'tier' : tier, 'rank': division, 'leaguePoints': Leaguepoints}
    relative_LP = normalization_rank(info)
    if champion_name == 'wukong':
        champion_name = 'monkeyking'
    champ_id = champ_Id[champion_name]
    dic = {'relative_LP' : relative_LP, 'champ_id': champ_id, 'champion_name': champion_name}
    return dic


def creation_match(blue_team, red_team):
    res = []
    for team in ([blue_team, red_team]):
        result = []
        result += [team['top']['relative_LP']]
        result += [team['jungle']['relative_LP']]
        result += [team['mid']['relative_LP']]
        result += [team['bottom']['relative_LP']]
        result += [team['utility']['relative_LP']]
        
        tab = number_of_champions * [0]
        positions = ['top', 'jungle', 'mid', 'bottom','utility']
        for pos in positions:
            tab[team[pos]['champ_id']] = 1 

        result += tab
        res.append(result)
    return res


def predict_winner_interface(tier_top_blue, div_top_blue, lp_top_blue, champ_top_blue, tier_jungle_blue, div_jungle_blue,
                             lp_jungle_blue, champ_jungle_blue, tier_mid_blue, div_mid_blue, lp_mid_blue, champ_mid_blue, 
                             tier_bottom_blue, div_bottom_blue, lp_bottom_blue, champ_bottom_blue, tier_utility_blue, 
                             div_utility_blue, lp_utility_blue, champ_utility_blue, tier_top_red, div_top_red, lp_top_red, 
                             champ_top_red, tier_jungle_red, div_jungle_red, lp_jungle_red, champ_jungle_red, tier_mid_red, 
                             div_mid_red, lp_mid_red, champ_mid_red, tier_bottom_red, div_bottom_red, lp_bottom_red, 
                             champ_bottom_red, tier_utility_red, div_utility_red, lp_utility_red, champ_utility_red         
                            ):
    all_champs = [champ_top_blue, champ_jungle_blue, champ_mid_blue, champ_bottom_blue, champ_utility_blue, champ_top_red,
                 champ_jungle_red, champ_mid_red, champ_bottom_red, champ_utility_red]
    for champ in all_champs:
      if all_champs.count(champ) > 1:
        sys.exit('error, a champ can only be choosen once')
            
    team_1 = {'top' : define_player(tier_top_blue, div_top_blue, lp_top_blue, champ_top_blue),
              'jungle' : define_player(tier_jungle_blue, div_jungle_blue, lp_jungle_blue, champ_jungle_blue),
          'mid' : define_player(tier_mid_blue, div_mid_blue, lp_mid_blue, champ_mid_blue), 
              'bottom' : define_player(tier_bottom_blue, div_bottom_blue, lp_bottom_blue, champ_bottom_blue),
          'utility' : define_player(tier_utility_blue, div_utility_blue, lp_utility_blue, champ_utility_blue)}

    team_2 = {'top' : define_player(tier_top_red, div_top_red, lp_top_red, champ_top_red),
              'jungle' : define_player(tier_jungle_red, div_jungle_red, lp_jungle_red, champ_jungle_red),
          'mid' : define_player(tier_mid_red, div_mid_red, lp_mid_red, champ_mid_red), 
              'bottom' : define_player(tier_bottom_red, div_bottom_red, lp_bottom_red, champ_bottom_red),
          'utility' : define_player(tier_utility_red, div_utility_red, lp_utility_red, champ_utility_red)}
    
    game = creation_match(team_1, team_2)
    result = model_to_use.predict([game])[0][0]
    winner = round(result)
    if winner == 1:
        equipe_winner = 'red'
    if winner == 0:
        equipe_winner = 'blue'
        result = 1 - result
    text = f'the {equipe_winner} team will win: final pourcentage = {100 * result}%'
    return text
    
    
all_tiers = ['IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'DIAMOND', 'MASTER' , 'GRANDMASTER', 'CHALLENGER']  
all_division = ['IV', 'III', 'II', 'I']


demo = gr.Interface(
    fn=predict_winner_interface,
    title="prediction of the winner",
    description="Predict which team (blue or red) will win the game",
    allow_flagging="never",
    inputs=[
        gr.inputs.Dropdown(choices= all_tiers, default="GOLD", label="tier top blue side"),
        gr.inputs.Dropdown(choices= all_division, default="I", label="division top blue side"),
        gr.inputs.Slider(minimum=0,maximum=2000,default=50,step=1, label="League points top blue side"),
        gr.inputs.Dropdown(choices= list(champ_Id.keys()), default="jax", label="champion top blue side"),
        
        gr.inputs.Dropdown(choices= all_tiers, default="GOLD", label="tier jungle blue side"),
        gr.inputs.Dropdown(choices= all_division , default="I", label="division jungle blue side"),
        gr.inputs.Slider(minimum=0,maximum=2000,default=50,step=1, label="League points jungle blue side"),
        gr.inputs.Dropdown(choices= list(champ_Id.keys()), default="elise", label="champion jungle blue side"),
        
        gr.inputs.Dropdown(choices= all_tiers, default="GOLD", label="tier mid blue side"),
        gr.inputs.Dropdown(choices= all_division , default="I", label="division mid blue side"),
        gr.inputs.Slider(minimum=0,maximum=2000,default=50,step=1, label="League points mid blue side"),
        gr.inputs.Dropdown(choices= list(champ_Id.keys()), default="fizz", label="champion mid blue side"),
        
        gr.inputs.Dropdown(choices= all_tiers, default="GOLD", label="tier bottom blue side"),
        gr.inputs.Dropdown(choices= all_division , default="I", label="division bottom blue side"),
        gr.inputs.Slider(minimum=0,maximum=2000,default=50,step=1, label="League points bottom blue side"),
        gr.inputs.Dropdown(choices= list(champ_Id.keys()), default="jhin", label="champion bottom blue side"),
        
        gr.inputs.Dropdown(choices= all_tiers, default="GOLD", label="tier utility blue side"),
        gr.inputs.Dropdown(choices= all_division , default="I", label="division utility blue side"),
        gr.inputs.Slider(minimum=0,maximum=2000,default=50,step=1, label="League points utility blue side"),
        gr.inputs.Dropdown(choices= list(champ_Id.keys()), default="bard", label="champion utility blue side"),
        
        
        gr.inputs.Dropdown(choices= all_tiers, default="GOLD", label="tier top red side"),
        gr.inputs.Dropdown(choices= all_division ,default="I", label="division top red side"),
        gr.inputs.Slider(minimum=0,maximum=2000,default=50,step=1, label="League points top red side"),
        gr.inputs.Dropdown(choices= list(champ_Id.keys()), default="darius", label="champion top red side"),
        
        gr.inputs.Dropdown(choices= all_tiers, default="GOLD", label="tier jungle red side"),
        gr.inputs.Dropdown(choices= all_division ,default="I", label="division jungle red side"),
        gr.inputs.Slider(minimum=0,maximum=2000,default=50,step=1, label="League points jungle red side"),
        gr.inputs.Dropdown(choices= list(champ_Id.keys()), default="hecarim", label="champion jungle red side"),
        
        gr.inputs.Dropdown(choices= all_tiers, default="GOLD", label="tier mid red side"),
        gr.inputs.Dropdown(choices= all_division , default="I", label="division mid red side"),
        gr.inputs.Slider(minimum=0,maximum=2000,default=50,step=1, label="League points mid red side"),
        gr.inputs.Dropdown(choices= list(champ_Id.keys()), default="yasuo", label="champion mid red side"),
        
        gr.inputs.Dropdown(choices= all_tiers, default="GOLD", label="tier bottom red side"),
        gr.inputs.Dropdown(choices= all_division , default="I", label="bottom utility red side"),
        gr.inputs.Slider(minimum=0,maximum=2000,default=50,step=1, label="League points bottom red side"),
        gr.inputs.Dropdown(choices= list(champ_Id.keys()), default="ashe", label="champion bottom red side"),
        
        gr.inputs.Dropdown(choices= all_tiers, default="GOLD", label="tier utility red side"),
        gr.inputs.Dropdown(choices= all_division , default="I", label="division utility red side"),
        gr.inputs.Slider(minimum=0,maximum=2000,default=50,step=1, label="League points utility red side"),
        gr.inputs.Dropdown(choices=list(champ_Id.keys()), default="alistar", label="champion utility red side"),

        ],
    outputs = 'text')


demo.launch(share=False, show_error = True)
