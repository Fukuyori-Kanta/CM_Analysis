import os
import ast
from collections import Counter
import numpy as np
import re

from utils.init_setting import setup_logger
from utils.file_io import read_csv, read_favo, write_ranking_data, write_csv, create_dest_folder

# ログ設定
logger = setup_logger(__name__)

def shaping_scene_data(scene_data):
    """シーンデータを整形して返す関数

    Parameters
    ----------
    scene_data : list
        シーンデータ

    Returns
    -------
    scene_dic
        辞書化したシーンデータ
        {video_id : [scene_no, start, end, label], [scene_no, start, end, label], ...}
    """
    scene_dic = {}              # 動画IDごとにシーンデータを辞書化
    prev_id = scene_data[0][0]  # 前データのID
    values = []
    for data in scene_data:
        video_id = data[0]      # 動画ID
        scene_no = data[1]      # シーン番号
        start = int(data[2])    # スタートフレーム
        end = int(data[3])      # エンドフレーム
        label = ast.literal_eval(data[4])  # ラベルのリスト

        # 前データのIDと動画IDが異なる場合
        if prev_id != video_id:
            scene_dic[prev_id] = values
            prev_id = video_id
            values = []
        else:
            values.append([scene_no, start, end, label])

        # 最後のデータ
        if scene_data.index(data) == len(scene_data)-1:
            scene_dic[prev_id] = values
    
    return scene_dic

def get_best_favo(start, end, favo_list):
    """フレーム情報から一番最適な好感度を返す関数
    
    [手順]
        1. フレーム範囲から一番近い秒数範囲を算出
        2. 秒数範囲に該当する好感度の平均を取る

    Parameters
    ----------
    start : int
        開始フレーム
    
    end : int
        終了フレーム

    favo_list : list
        好感度のリスト（毎秒ごと）

    Returns
    -------
    float
        フレーム範囲付近の好感度の平均値
    """ 
    max_sec = len(favo_list)    # 動画秒数
    fps = 30    # フレームレート
    sec_list = [i*fps for i in range(1, max_sec+1)] # 秒数（フレーム表記）のリスト
    start_near_sec = sec_list[np.abs(np.asarray(sec_list) - start).argmin()]    # 一番近い秒数
    end_near_sec = sec_list[np.abs(np.asarray(sec_list) - end).argmin()]
    near_sec_range =  [start_near_sec+i*fps for i in range((end_near_sec-start_near_sec)//fps+1)]    # 一番近い秒数範囲
    candidate_favo = [favo_list[ns//30-1] for ns in near_sec_range] # 該当好感度

    # 好感度の平均値を返す
    return round(sum(candidate_favo) / len(candidate_favo), 3)

def add_favo_to_scene(scene_data, favo_data):
    """シーンデータに好感度データを付与して返す関数
    
    Parameters
    ----------
    scene_data : list
        シーンデータ
    
    favo_data : list
        好感度データ

    Returns
    -------
    scene_favo_list : list
        場面データ [[video_id, scene_no, start, end, [label, ...], best_favo], ....]
    """ 
    header, *favo_data = favo_data  # ヘッダーと分離
    video_id_col = header.index('映像コード')  # 動画IDの列
    video_id_list = [data[video_id_col] for data in favo_data]  # 動画IDリスト

    # データの整形（動画IDごとに辞書化）
    favo_dic = {data[video_id_col] : [float(favo) for favo in data[video_id_col+1:] if favo != '']for data in favo_data}    # 好感度データ
    scene_dic = shaping_scene_data(scene_data)  # シーンデータ

    # 各シーンに好感度を付与
    scene_favo_list = [] # シーンのリスト
    for video_id in video_id_list:
        for sl in scene_dic[video_id]:
            scene_no = sl[0]    # シーン番号
            start = sl[1]       # スタートフレーム
            end = sl[2]         # エンドフレーム
            labels = sl[3]      # ラベル

            # シーン範囲から最適な好感度を取得
            best_favo = get_best_favo(start, end, favo_dic[video_id])

            scene_favo_list.append([video_id, scene_no, start, end, labels, best_favo])

    return scene_favo_list

def favo_analysis(cmData_top_path, cmData_btm_path, scene_dir, favo_dir):
    """ラベル件数を集計して出力する関数

    [手順]
        1. データの読み込み・前処理
        2. 場面データに好感度データの追加
        3. ラベル件数のカウント
        4. 結果の保存

    Parameters
    ----------
    cmData_top_path : str
        CMデータ(上位1000)のファイルパス

    cmData_btm_path : str
        CMデータ(下位1000)のファイルパス
    
    scene_dir : str
        シーン(動画)の保存フォルダパス

    favo_dir :str
        好感度の結果を出力するフォルダパス
    """
    # CMデータの取得
    cmData_top = read_csv(cmData_top_path) # 上位データ
    cmData_btm = read_csv(cmData_btm_path) # 下位データ

    # 場面データの読み込み
    scene_data = read_csv(scene_dir, needs_skip_header=True)
    
    # 場面データに好感度データを追加
    top_scene_list = add_favo_to_scene(scene_data, cmData_top)
    btm_scene_list = add_favo_to_scene(scene_data, cmData_btm)

    # ラベル件数カウント
    # 上位データ
    top_all_label = [label for all_scene in top_scene_list for label in all_scene[4]] 
    top_label_cnt = [[c[0], int(c[1])] for c in Counter(top_all_label).most_common()]
    top_verb_label = [data for data in top_all_label if data[0].islower()]
    top_verb_cnt = [[c[0], int(c[1])] for c in Counter(top_verb_label).most_common()]
    result_txt = (
        f'シーン件数: {len(top_scene_list)}, ラベル件数: {len(top_all_label)}(動作ラベル: {len(top_verb_cnt)}), '
        f'総秒数: {sum([int(data[3]) for data, n_data in zip(top_scene_list, top_scene_list[1:]) if n_data[2] == 0] + [top_scene_list[-1][3]]) // 30} 秒'
    )
    logger.debug(result_txt)

    # 下位データ
    btm_all_label = [label for all_scene in btm_scene_list for label in all_scene[4]] 
    btm_label_cnt = [[c[0], int(c[1])] for c in Counter(btm_all_label).most_common()]
    btm_verb_label = [data for data in btm_all_label if data[0].islower()]
    btm_verb_cnt = [[c[0], int(c[1])] for c in Counter(btm_verb_label).most_common()]
    result_txt = (
        f'シーン件数: {len(btm_scene_list)}, ラベル件数: {len(btm_all_label)}(動作ラベル: {len(btm_verb_cnt)}),'
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
    write_csv(top_verb_cnt, os.path.normpath(os.path.join(favo_dir, 'top_verb.csv')))
    write_csv(btm_verb_cnt, os.path.normpath(os.path.join(favo_dir, 'btm_verb.csv')))
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