import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns

import os
from logging import getLogger, StreamHandler, DEBUG

from setting import setup_path
from utils.file_io import read_csv, write_csv
from cut_segmentation_mod import read_video_data, cut_point_detect

def str2int(string):
    """
    文字列になっているリストから数字のリストに変換する関数
    ex.)  '[29, 64, 94, 156]' →  [29, 64, 94, 156]

    Parameters
    ----------
    string : String
        文字列になっているリスト

    Returns
    -------
    num_list : list
        数字に変換したリスト  
    """
    import re
    
    regex = re.compile('\d+') # 文字列から1つ以上の数字にマッチするパターン

    # 文字列からリストへ
    for line in string.splitlines():
        string = regex.findall(line)

    # リスト内の文字列数字を数字へ
    num_list = [int(s) for s in string]

    return num_list

def data_preprocessing(cut_point):
    """
    データ分析用の前処理を行う関数
    カット点の数値を１、それ以外を０としたリストを作成

    Parameters
    ----------
    cut_point : list
        カット検出点（フレーム番号）のリスト 

    Returns
    -------
    list : data
        分割該当のフレームを１、それ以外を０としたリスト
    """
    data = [0] * (cut_point[-1] + 1) # ０で初期化
    for f in cut_point:
        data[f] = 1 # 該当フレームを１にする

    return data

def cut_evaluate(video_id, y_true, y_pred):
    """
    評価指標を算出する関数
    正答率、適合率、再現率、F値を算出して返す
    
    ※y_true, y_pred の例
        0 : 陰性　1: 陽性 
        y_true = [1, 0, 0, 1] # 正解
        y_pred = [1, 0, 0, 1] # 予測

    Parameters
    ----------
    video_id : str
        動画ID 
    
    y_ture : lsit
        正解のカット点

    y_pred : list
        検出したカット点

    Returns
    -------
    float, float, float, float, list[tn, fp, fn, tp]
        それぞれ 正答率、再現率、適合率、F値、混同行列
    """
    cm = confusion_matrix(y_true, y_pred)       # 混同行列

    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred), precision_score(y_true, y_pred), f1_score(y_true, y_pred), cm.flatten()

def result_graph_show(result_data):
    """
    TODO 
    現在はリファクタリングしないが、
    今後修正するかも
    """
    # [個々のCMの結果をヒストグラム化]------------------------------------------------------------
    #figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
    fig = plt.figure()

    #add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    acc = np.array(result_accuracy)
    re = np.array(result_recall)
    pre = np.array(result_precision)
    f1 = np.array(result_f1)

    c1,c2,c3,c4 = "blue","green","red","black"  # 各プロットの色

    ax1.hist(acc, color=c1)
    ax1.set_title('accuracy')
    ax2.hist(re, color=c2)
    ax2.set_title('recall')
    ax3.hist(pre, color=c3)
    ax3.set_title('precision')
    ax4.hist(f1, color=c4)
    ax4.set_title('f1')
    fig.tight_layout()  #レイアウトの設定
    plt.show()
    # -----------------------------------------------------------------
    # [CM全体の混合行列]
    print('TP = ' + str(tp_cnt))
    print('FN = ' + str(fn_cnt))
    print('FP = ' + str(fp_cnt))
    print('TN = ' + str(tn_cnt))

    precision_all = tp_cnt / (tp_cnt + fp_cnt) * 100    # 全体の適合率
    recall_all = tp_cnt / (tp_cnt + fn_cnt) * 100       # 全体の再現率
    f1_all = 2 * precision_all * recall_all / (precision_all + recall_all)  # 全体のF値
    print('適合率 = ' + str(round(precision_all, 2)) + ' %')
    print('再現率 = ' + str(round(recall_all, 2)) + ' %')
    print('F値 = ' + str(round(f1_all, 2)) + ' %')

    cm = np.matrix([[tn_cnt, fp_cnt], [fn_cnt, tp_cnt]])
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.show()

if __name__ == '__main__':
    # --------------------------------------------------
    # 各種設定
    # --------------------------------------------------
    os.chdir(r'C:\Users\hukuyori\CM_Analysis')  # TODO 後で消す

    # path の設定
    path = setup_path()   
    root_path, video_path, cmData_path, result_cut_path, result_cut_img_path, ansData_path, result_cut_eva_path = path 

    # ログ設定
    logger = getLogger(__name__)
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    # --------------------------------------------------
    # 動画IDの取得
    # --------------------------------------------------
    cm_data = read_csv(cmData_path) # CMデータ（動画ID, 企業名, 商品名, 作品名, クラスタパターン）
    video_id_list = [cm_data[i][1] for i in range(len(cm_data))]    # 動画IDリストを作成
    """
    # 実写以外のCMは判定しない
    delete_list = ['D191032200', 'A191025316', 'L191051119', 'A201054575']  # 削除のリスト
    video_id_list = set(video_id_list) ^ set(delete_list)
    """
    # --------------------------------------------------
    # 正解データの取得
    # --------------------------------------------------
    answer_data = read_csv(ansData_path)   # 正解データの取得 
    answer_data = {ans[0]:ans[1] for ans in answer_data}    # 正解データを辞書化

    # --------------------------------------------------
    # 変数の初期化・宣言
    # --------------------------------------------------
    # 評価指標
    result_accuracy = []    # 正答率の結果を格納
    result_recall = []      # 再現率の結果を格納
    result_precision = []   # 適合率の結果を格納
    result_f1 = []          # F値の結果を格納

    # 各種カウンタ
    tp_cnt = 0
    fn_cnt = 0
    fp_cnt = 0
    tn_cnt = 0
    correct_cnt = 0

    # 不正解の動画IDを格納
    incorrect_id = []

    bad_f1 = []
    # --------------------------------------------------
    # カット点の評価
    # --------------------------------------------------
    for video_id in video_id_list:
        input_video_path = video_path + '\\' + video_id + '.mp4' # 動画ファイルの入力パス 
        
        # --------------------------------------------------
        # 動画の読み込み、フレームデータと動画情報を抽出
        # --------------------------------------------------
        frames, video_info = read_video_data(input_video_path)
        
        # --------------------------------------------------
        # カット点の検出
        # --------------------------------------------------
        cut_point = cut_point_detect(frames)

        # --------------------------------------------------
        # データの前処理
        # --------------------------------------------------
        # 文字列になっているリストを数字入りリストに変換
        answer_data[video_id] = str2int(answer_data[video_id])  
        
        # カット点と正解データの総フレーム数が合わないとエラーが出るため、総フレーム数を一致させる処理
        # 一致しない理由は、動画データ読み取り時に最後のフレームまで読み込まない場合がある（不具合?）
        if answer_data[video_id][-1] != cut_point[-1]:
            cut_point[-1] = answer_data[video_id][-1]

        y_true = data_preprocessing(answer_data[video_id])  # 正解データ 
        y_pred = data_preprocessing(cut_point)              # 予測データ
        
        # --------------------------------------------------
        # 各評価指標の算出・集計
        # --------------------------------------------------
        logger.debug(video_id + ' は以下の評価です。')

        accuracy, recall, precision, f1, cm = cut_evaluate(video_id, y_true, y_pred)

        print('正答率 = ' + str(round(accuracy, 2)))
        print('再現率 = ' + str(round(recall, 2)))
        print('適合率 = ' + str(round(precision, 2)))
        print('F値 = ' + str(round(f1, 2)))

        logger.debug('------------------------')
        
        # 全体の評価用に格納
        result_accuracy.append(accuracy)
        result_recall.append(recall)
        result_precision.append(precision)
        result_f1.append(f1)

        # 全体の混合行列を作成するためカウント
        tn, fp, fn, tp = cm
        tp_cnt += tp
        fn_cnt += fn
        fp_cnt += fp
        tn_cnt += tn

        # 完璧に分割出来たCMの個数をカウント
        # 正答率が１の時 = そのCMの分割は完璧
        if accuracy == 1:
            correct_cnt += 1
        else:
            incorrect_id.append(video_id)

        if f1 <= 0.5:
            bad_f1.append(video_id)

    # --------------------------------------------------
    # CM全体の結果の出力
    # --------------------------------------------------
    print('------CM全体の結果------')
    # 混同行列の各成分を表示
    print('TP = ' + str(tp_cnt))
    print('FN = ' + str(fn_cnt))
    print('FP = ' + str(fp_cnt))
    print('TN = ' + str(tn_cnt))
    
    # CM全体での評価を表示
    recall_all = tp_cnt / (tp_cnt + fn_cnt) * 100       # 全体の再現率
    precision_all = tp_cnt / (tp_cnt + fp_cnt) * 100    # 全体の適合率
    f1_all = 2 * precision_all * recall_all / (precision_all + recall_all)  # 全体のF値

    print('適合率 = ' + str(round(precision_all, 2)) + ' %')
    print('再現率 = ' + str(round(recall_all, 2)) + ' %')
    print('F値 = ' + str(round(f1_all, 2)) + ' %')
    print("正答率 = " + str(round(correct_cnt / len(video_id_list) * 100, 2)) + ' %')

    # 不正解のCMを表示
    print(incorrect_id)

    # 特に間違えているCMを表示
    print(bad_f1)

    # カット数一覧をCSVに書き出し
    result_cut_evaluates = [video_id_list, result_recall, result_precision, result_f1]
    write_csv(result_cut_evaluates, result_cut_eva_path)

    logger.debug('全動画の評価が終了しました。')
    logger.debug('-' * 90)