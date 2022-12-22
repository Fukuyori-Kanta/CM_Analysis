import os
import ast
from collections import Counter
import heapq
import numpy as np
import re
import csv

from utils.init_setting import setup_logger
from utils.file_io import read_csv, read_favo, write_ranking_data, write_csv, create_dest_folder

# ログ設定
logger = setup_logger(__name__)

def order_desc(dic, N):
    # valueでソートしたときの上位N個
    lists = heapq.nlargest(N, dic.items(), key=lambda x: x[1])

    new_dict = {}
    for l in lists:
        new_dict[l[0]] = l[1]
    
    return new_dict

def shaping_favo_data(scene_data, favo_data, video_id_list):
    """
    好感度データを整形して返す関数
    """
    # データの整形（動画IDをキーとした辞書の作成）
    # シーンデータの整形
    # {video_id : [scene_no, start, end, label], [scene_no, start, end, label], ...}
    prev_id = scene_data[0][0]   # 前データのID
    scene_dic = {}  # 動画IDごとにシーンデータを辞書化
    l = []
    for data in scene_data:
        video_id = data[0]      # 動画ID
        scene_no = data[1]      # シーン番号
        start = int(data[2])    # スタートフレーム
        end = int(data[3])      # エンドフレーム
        label = ast.literal_eval(data[4])  # ラベルのリスト

        # 前デーのIDと動画IDが違うとき
        if prev_id != video_id:
            scene_dic[prev_id] = l  # 
            prev_id = video_id
            l = []
        
        l.append([scene_no, start, end, label])

        # 最後のデータ
        if scene_data.index(data) == len(scene_data)-1:
            scene_dic[prev_id] = l
    
    # 好感度データの整形
    # {video_id : [sec, favo], [sec, favo], ...}
    favo_dic = {}   # 動画IDごとに好感度データを辞書化
    for data in favo_data:
        video_id = data[49]     # 動画ID
        favo_dic[video_id] = [[int(idx+1), float(favo)] for idx, favo in enumerate(data[50:]) if favo != '']
    
    return scene_dic, favo_dic

def get_best_favo(start, end, favo_list):
    """
    シーン範囲に近い秒数の好感度を返す関数
    """
    max_sec = len(favo_list)
    sec_list = [i*30 for i in range(1, max_sec+1)]
    start_near_sec = sec_list[np.abs(np.asarray(sec_list) - start).argmin()]
    end_near_sec = sec_list[np.abs(np.asarray(sec_list) - end).argmin()]
    near_sec_list =  [start_near_sec+i*30 for i in range((end_near_sec-start_near_sec)//30+1)]
    candidate_favo = [favo_list[ns//30-1][1] for ns in near_sec_list]
    
    # 好感度の平均値を返す
    return round(sum(candidate_favo) / len(candidate_favo), 3)

def ranking(scene_favo_dic, scene_dic, i, result_path):
    # 上位、中位、下位で〇〇シーンに付与されたラベル上位10件を出力
    dic_len = len(scene_favo_dic)   # 総シーン数
    all_favo = order_desc(scene_favo_dic, dic_len)   # 好感度の降順
    scene_range = 1000  # 1000件のシーン範囲

    all_scene = []  # 全シーンのデータ [動画名, シーン番号, 好感度, ラベルのリスト]
    for scene in all_favo.keys():
        video_id = scene[0]
        scene_no = scene[1]
        favo_val = all_favo[scene]
        
        # 該当のラベルデータ読み出し
        label = [sd[3] for sd in scene_dic[video_id] if sd[0] == scene_no][0]
        
        all_scene.append([video_id, scene_no, favo_val, label])
    
    all_label = []  # 全ラベルのデータ
    for scene in all_scene:
        label = [re.sub("\(.*\)", "", s) for s in scene[3]] # 個数削除
        all_label += label
        
    # 総ラベル件数辞書 {'女性' : 3112, '男性' : 2362, ...}
    all_cnt = [[c[0], str(c[1])] for c in Counter(all_label).most_common()]
    label_cnt_dic = {a[0]:int(a[1]) for a in all_cnt}
    
    # 上位
    top_scene = all_scene[:scene_range]
    top_label = []
    for scene in top_scene:
        label = [re.sub("\(.*\)", "", s) for s in scene[3]] # 個数削除
        top_label += label

    # 件数順にソート 
    top_cnt = [[c[0], c[1], str(round(c[1] / label_cnt_dic[c[0]], 2))] for c in Counter(top_label).most_common()]
    
    for t in top_cnt[:10]:
        print(t)
    print('-----------')
    
    # 中位
    mid_scene = all_scene[(dic_len//2)-(scene_range//2):(dic_len//2)+(scene_range//2)]
    mid_label = []
    for scene in mid_scene:
        label = [re.sub("\(.*\)", "", s) for s in scene[3]] # 個数削除
        mid_label += label
    
    # 件数順にソート
    mid_cnt = [[c[0], c[1], str(round(c[1] / label_cnt_dic[c[0]], 2))]   for c in Counter(mid_label).most_common()]

    for m in mid_cnt[:10]:
        print(m)
    print('-----------')

    # 下位
    btm_scene = all_scene[-scene_range:]
    btm_label = []
    for scene in btm_scene:
        label = [re.sub("\(.*\)", "", s) for s in scene[3]] # 個数削除
        btm_label += label

    # 件数順にソート
    btm_cnt = [[c[0], c[1], str(round(c[1] / label_cnt_dic[c[0]], 2))]   for c in Counter(btm_label).most_common()]

    for b in btm_cnt[:10]:
        print(b)
    
    # CSVファイルに出力
    data = [top_cnt[:10], mid_cnt[:10], btm_cnt[:10]]
    for num in range(len(data)):
        if i == 0:
            file_name = 'facter_' + str(i) + '.csv'
        elif i == 1:
            file_name = 'facter_A.csv'
        else:
            file_name = 'facter_' + str(i-1) + '.csv'
        
        output_path = os.path.normpath(os.path.join(result_path, file_name))
        write_ranking_data(data[num], output_path, num)

def add_favo_to_scene(scene_data, favo_data):
    video_id_list = [data[49] for data in favo_data]

    # データの整形
    scene_dic, favo_dic = shaping_favo_data(scene_data, favo_data, video_id_list)
    
    # 各シーンに好感度を付与
    scene_favo_list = [] # シーンのリスト
    for video_id in video_id_list:
        favo_list = favo_dic[video_id]
        scene_list = scene_dic[video_id]

        for sl in scene_list:
            scene_no = sl[0]    # シーン番号
            start = sl[1]       # スタートフレーム
            end = sl[2]         # エンドフレーム
            labels = sl[3]      # ラベル

            # シーン範囲から最適な好感度を取得
            best_favo = get_best_favo(start, end, favo_list)

            scene_favo_list.append([video_id, scene_no, start, end, labels, best_favo])

    return scene_favo_list

def get_label_count(scene_list):
    print(scene_list[:10])
    all_label = []  # 全ラベルのデータ
    for all_scene in scene_list:
        label = [s for s in all_scene[4]] # 個数削除
        all_label += label
    
    # 総ラベル件数辞書 {'女性' : 3112, '男性' : 2362, ...}
    label_cnt = [[c[0], int(c[1])] for c in Counter(all_label).most_common()]

    verv = [data for data in all_label if data[0].islower()]    # 
    #top_label_cnt_verv = [[c[0], int(c[1])] for c in Counter(verv).most_common()]
    # write_csv(top_label_cnt_verv, r'result\top_verv_labels.csv')

    print(f'シーン件数: {len(scene_list)}, ラベル件数: {len(all_label)}(動作ラベル: {len(verv)}), 総秒数: {sum([int(data[3]) for data, n_data in zip(scene_list, scene_list[1:]) if n_data[2] == 0] + [scene_list[-1][3]]) // 30} 秒')

    return label_cnt

def favo_analysis(cmData_top_path, cmData_btm_path, scene_path, favo_dir):
    # CMデータの取得
    cmData_top = read_csv(cmData_top_path, needs_skip_header=True) # 上位データ
    cmData_btm = read_csv(cmData_btm_path, needs_skip_header=True) # 下位データ
    
    # 場面データの読み込み
    scene_data = read_csv(scene_path, True)
    
    # 場面データに好感度データを追加
    top_scene_list = add_favo_to_scene(scene_data, cmData_top)
    btm_scene_list = add_favo_to_scene(scene_data, cmData_btm)

    # ラベル件数カウント
    # 上位データ
    top_all_label = [label for all_scene in top_scene_list for label in all_scene[4]] 
    top_label_cnt = [[c[0], int(c[1])] for c in Counter(top_all_label).most_common()]
    top_verv_label = [data for data in top_all_label if data[0].islower()]
    top_verv_cnt = [[c[0], int(c[1])] for c in Counter(top_verv_label).most_common()]
    result_txt = (
        f'シーン件数: {len(top_scene_list)}, ラベル件数: {len(top_all_label)}(動作ラベル: {len(top_verv_cnt)}), '
        f'総秒数: {sum([int(data[3]) for data, n_data in zip(top_scene_list, top_scene_list[1:]) if n_data[2] == 0] + [top_scene_list[-1][3]]) // 30} 秒'
    )
    logger.debug(result_txt)

    # 下位データ
    btm_all_label = [label for all_scene in btm_scene_list for label in all_scene[4]] 
    btm_label_cnt = [[c[0], int(c[1])] for c in Counter(btm_all_label).most_common()]
    btm_verv_label = [data for data in btm_all_label if data[0].islower()]
    btm_verv_cnt = [[c[0], int(c[1])] for c in Counter(btm_verv_label).most_common()]
    result_txt = (
        f'シーン件数: {len(btm_scene_list)}, ラベル件数: {len(btm_all_label)}(動作ラベル: {len(btm_verv_cnt)}),'
        f'総秒数: {sum([int(data[3]) for data, n_data in zip(btm_scene_list, btm_scene_list[1:]) if n_data[2] == 0] + [btm_scene_list[-1][3]]) // 30} 秒'
    )
    logger.debug(result_txt)

    # 結果フォルダを作成
    create_dest_folder(favo_dir)

    # CSVファイルに保存
    write_csv(top_scene_list, os.path.normpath(os.path.join(favo_dir, 'top_scene_data.csv')))
    write_csv(btm_scene_list, os.path.normpath(os.path.join(favo_dir, 'btm_scene_data.csv')))
    write_csv(top_label_cnt, os.path.normpath(os.path.join(favo_dir, 'top_labels.csv')))
    write_csv(btm_label_cnt, os.path.normpath(os.path.join(favo_dir, 'btm_labels.csv')))
    write_csv(top_verv_cnt, os.path.normpath(os.path.join(favo_dir, 'top_verv.csv')))
    write_csv(btm_verv_cnt, os.path.normpath(os.path.join(favo_dir, 'btm_verv.csv')))