import os
import ast
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from utils.init_setting import setup_logger
from utils.file_io import read_csv, write_csv, create_dest_folder
from utils.cut_segmentation_mod import read_video_data

INTEGRATION_THRESHOLD = 0.93    # シーンにする際の閾値
EXTENSION = '.mp4'              # 保存するシーンの拡張子（MP4）

# ログ設定
logger = setup_logger(__name__)

def two_doc_2_vec(documents):
    """２つの文書をベクトル化する関数

    Parameters
    ----------
    documents : list
        ２つ文書のリスト

    Returns
    -------
    numpy.ndarray
        ２つの文書のベクトル
    """
    np.set_printoptions(precision=2)
    docs = np.array(documents)
    vectorizer = TfidfVectorizer(use_idf=False, token_pattern=u'(?u)\\b\\w+\\b')
    vecs = vectorizer.fit_transform(docs)

    return vecs.toarray()

def calc_similarity(list_1, list_2):
    """２つのリストの類似度を算出する関数

    Parameters
    ----------
    list_1 : list
        １つめのリスト

    list_2 : list
        ２つめのリスト

    Returns
    -------
    similarity : numpy.float64
        ２つのリストの類似度
    """
    docs = [' '.join(list_1), ' '.join(list_2)] # ２つの文書にする
    cs_array = np.round(cosine_similarity(two_doc_2_vec(docs), two_doc_2_vec(docs)), 3) # ２つの文書から類似行列を算出
    similarity = cs_array[0][1] # 類似行列から類似度を抽出

    return similarity

def judge_whether_integrate(id, next_id, labels, next_labels):
    """ラベル同士の類似度から、統合するかどうかを判定する関数

    Parameters
    ----------
    id : str
        判定用のID

    next_id : str
        次データのID

    labels : list
        ラベルのリスト
    
    next_labels : lsit
        次データのラベルリスト

    Returns
    -------
    bool
        比較結果（統合するかどうか）
    """
    # 両方のラベルデータが存在する場合
    if labels and next_labels:  
        # 両ラベルの類似度算出
        similarity = calc_similarity(labels, next_labels)

        # 類似度が閾値以上の場合
        if similarity >= INTEGRATION_THRESHOLD and id == next_id:
            return True
        else:
            return False
    else:
        return False

def create_cut_data(labels, cut_point_dict):
    """ラベルデータとカット点データからカットデータを作成する関数

    Parameters
    ----------
    labels : list
        ラベルデータ

    cut_point_dict : dict
        カット点データ

    Returns
    -------
    cut_data : list
        カットデータ    ['動画ID', 'カット番号', 'スタートフレーム', 'エンドフレーム', '[ラベルのリスト]']
    """
    cut_data = []   # カットデータ
    for data in labels:
        video_id = data[0]  # 動画ID
        cut_no = data[1]    # カット番号
        labels = data[2]    # ラベル
        
        # 最初のカットの場合
        if cut_no == 1:
            start_frame = 0 # スタートフレーム
            end_frame = cut_point_dict[video_id][cut_no-1]  # エンドフレーム
        else:
            start_frame = cut_point_dict[video_id][cut_no-2] + 1
            end_frame = cut_point_dict[video_id][cut_no-1]
        
        cut_data.append([video_id, cut_no, start_frame, end_frame, labels])

    return cut_data

def create_scene_data(cut_data):
    """カットデータからシーンデータを作成する関数

    Parameters
    ----------
    cut_data : list
        カットデータ

    Returns
    -------
    scene_data : list
        シーンデータ    ['動画ID', 'シーン番号', 'スタートフレーム', 'エンドフレーム', '[ラベルのリスト]']
    """
    scene_data = [] # シーンデータ 
    for data, next_data in zip(cut_data, cut_data[1:]):
        video_id = data[0]              # 動画ID
        next_video_id = next_data[0]    # 次データの動画ID
        cut_no = data[1]                # カット番号
        cut_range = (data[2], data[3])  # カット範囲
        label = [labels[0] if isinstance(labels, list) else labels for labels in data[4]]           # ラベル
        next_label = [labels[0] if isinstance(labels, list) else labels for labels in next_data[4]] # 次データのラベル

        # ラベルの類似度から統合するかを判定
        isintegrate = judge_whether_integrate(video_id, next_video_id, label, next_label)
        
        if isintegrate:
            results_label = next_label
            cut_range = (data[2], next_data[3])  # カット範囲統合
        else:
            results_label = label
                
        # ラベルデータに追加
        scene_data.append([video_id, cut_no, cut_range[0], cut_range[1], results_label])

    # 最後のデータ追加
    scene_data.append(cut_data[-1])
    
    # 統合による削除対象
    remove_target = [n for p, n in zip(scene_data, scene_data[1:]) if p[0] == n[0] and int(p[3]) >= int(n[2])]
    for rt in remove_target:
        scene_data.remove(rt)

    # シーン番号の整理（削除により欠けている部分を修正）
    scene_no_list = []
    for i in range(len(scene_data)):
        # 最初のインデックスの場合
        if i == 0:
            scene_no = 1
        else:
            # 前データのidと同じ場合
            if scene_data[i][0] == scene_data[i-1][0]:
                scene_no += 1
            else:
                scene_no = 1

        scene_no_list.append(scene_no)
    
    for data, scene_no in zip(scene_data, scene_no_list):
        data[1] = scene_no

    return scene_data

def calc_scene_point(scene_data):
    """シーン分割点を計算する関数

    Parameters
    ----------
    scene_data : list
        シーンデータ

    Returns
    -------
    scene_point_dic : dict
        シーン分割点の辞書　{動画ID: エンドフレームのリスト}
    """
    current_id = scene_data[0][0]   # 現在の動画ID
    scene_point_dic = {}            # シーン分割点の辞書
    end_frames = []                 # エンドフレームのリスト
    for data in scene_data:
        video_id = data[0]  # 動画IDID
        end = int(data[3])  # エンドフレーム

        if current_id != video_id:
            scene_point_dic[current_id] = end_frames
            current_id = video_id
            end_frames = []
        
        end_frames.append(end)

        if scene_data.index(data) == len(scene_data)-1:
            scene_point_dic[current_id] = end_frames
    
    return scene_point_dic   

def save_scene(video_id_list, scene_point_dic, video_path, scene_path):
    """動画を分割して保存する関数

    Parameters
    ----------
    video_id_list : list
        動画IDのリスト

    scene_point_dic : dict
        シーン分割点（フレーム番号）の辞書 {動画ID: エンドフレームのリスト}
    
    video_path : str
        動画データが存在するフォルダパス

    scene_path : str
        シーン分割結果（動画）を保存するフォルダパス
    """
    # シーン保存先のフォルダを作成
    create_dest_folder(scene_path)

    # シーン保存
    for video_id in video_id_list:
        file_name = video_id + EXTENSION    # ファイル名.拡張子
        input_video_path = os.path.normpath(os.path.join(video_path, file_name)) # 動画ファイルの入力パス
        scene_point = scene_point_dic[video_id] # シーンの分割点

        # 保存先フォルダの作成
        dest_path = os.path.join(scene_path, video_id) # 各動画のカット分割結果の保存先
        create_dest_folder(dest_path)
        
        # 動画の読み込み
        frames, video_info = read_video_data(input_video_path)
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]   # RGBからBGRに戻す（戻さないと色が反転したまま保存される）

        fps, width, height = video_info # 動画情報（fps, 幅, 高さ）
        fourcc = 0x00000021             # 動画の保存形式(H264形式でエンコード)
        writer = []                     # 書き込み用のリスト
        begin = 0                       # シーン最初のフレーム

        scene_count = len(scene_point) # シーン数
        for i in range(scene_count):
            save_scene_path = os.path.normpath(os.path.join(dest_path, 'scene' + str(i+1) + EXTENSION))    # 保存先
            writer.append(cv2.VideoWriter(save_scene_path, fourcc, fps, (int(width), int(height))))
            for j in range(begin, scene_point[i]+1):
                writer[i].write(frames[j])
            begin = scene_point[i]+1

        logger.debug(video_id + '_scene1 ～ ' + str(scene_count) + 'を保存しました')
        logger.debug('保存先 : ' + dest_path)
        logger.debug('-' * 90)

def scene_integration(cut_point_path, label_path, video_path, scene_path, scene_data_path):
    """シーンに統合・保存する関数

    [手順]
        1. データの読み込み・前処理
        2. カットデータの作成
        3. シーンデータの作成
        4. シーンの保存
        5. シーンデータの保存

    Parameters
    ----------
    cut_point_path : str
        カット分割結果（カット点）を保存するファイルパス（.csv）
    
    label_path : str
        ラベルデータ（.csv）のファイルパス

    video_path : str
        動画データが存在するフォルダパス

    scene_path : str
        シーン（動画）の保存フォルダパス

    scene_data_path : str
        シーンのデータ（.csv）の保存ファイルパス
    """
    # 動画IDリスト、カット点データの読み込み
    video_id_list, cut_point_list = read_csv(cut_point_path)         # 動画IDリスト, カットデータ 
    cut_point_list = [ast.literal_eval(data) for data in cut_point_list]  # 整形
    
    # ラベルデータの読み込み
    labels = [[data[0], int(data[1]), ast.literal_eval(data[2])] for data in read_csv(label_path, needs_skip_header=True)]
    
    # カット点辞書の作成
    cut_point_dict = {video_id : cut_point for video_id, cut_point in zip(video_id_list, cut_point_list)}
    
    # カットデータの作成
    cut_data = create_cut_data(labels, cut_point_dict)
    
    # シーンデータの作成
    scene_data = create_scene_data(cut_data)
    
    # シーン分割点の辞書を作成
    scene_point_dic = calc_scene_point(scene_data)

    # シーンを動画として保存
    save_scene(video_id_list, scene_point_dic, video_path, scene_path)

    # シーンデータの保存
    field_name = ['動画ID', 'シーン番号', 'スタートフレーム', 'エンドフレーム', '[ラベルのリスト]']
    scene_data.insert(0, field_name)
    write_csv(scene_data, scene_data_path)