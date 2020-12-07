import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import csv
from logging import getLogger, StreamHandler, DEBUG

# 閾値の設定
CUT_THRESHOLD = 85          # カット分割時の閾値
CUT_BETWEEN_THRESHOLD = 90  # カット間フレームの削除時の閾値
HIST_THRESHOLD = 0.8        # ヒストグラムインタセクション（HI）の類似度比較時の閾値
FLASH_THRESHOLD = 0.65      # フラッシュ検出時のHIの類似度比較時の閾値
EFFECT_THRESHOLD = 0.8      # エフェクト検出時のHIの類似度比較時の閾値

# ログ設定
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

def dest_folder_create(dest_path):
    """
    保存先フォルダを作成する関数

    Parameters
    ----------
    dest_path : str
        保存先フォルダのパス
    """
    # 保存先フォルダの作成
    # 既に存在する場合は削除
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)    # フォルダ削除
    os.mkdir(dest_path)     # 保存先フォルダの作成

def MSE(diff): 
    """
    平均二乗誤差(MSE)を行って、結果を帰す関数

    diff = y[i+1] - y[i]    （差分画像）
    MSE = 1/n Σ (diff)^2    （平均二乗誤差）

    Parameters
    ----------
    diff : numpy.ndarray
        平均二乗誤差を求めたいデータ

    Returns
    -------
    numpy.float64
        カット検出点（フレーム番号）のリスト 
    """
    return np.mean(np.square(diff))

def target_delete(cut_point, delete_target):
    """
    カット点の修正時に、不要になったフレームを削除する関数

    Parameters
    ----------
    cut_point : list
        カット点のフレーム番号リスト

    delete_target : list
        不要なフレーム番号リスト

    Returns
    -------
    cut_point : list
        カット点のフレーム番号リスト（削除後）
    """
    delete_target = list(set(delete_target))    # 重複を排除

    # 削除対象のカット点を削除
    for i in delete_target:
        cut_point.remove(i)    # 削除
    
    return cut_point

def cut_between_frame_delete(cut_point, diff_images):
    """
    カット間フレームを削除する関数
    カット間フレーム = カットとカットに稀出来る不要なフレーム

    [削除の理由]
        カット点と誤検出している状態のため削除する必要がある
    [方法]
        必ず正解のカット点の1つ後ろのフレームに出来るため、連続なカット点を探る（削除候補）
        カット間フレームの前後のフレームの差分画像をMSEにかける
        その結果が閾値以下（ほとんど同じ）ならカット間フレームであり、削除対象
        カット間フレームを削除

    Parameters
    ----------
    cut_point : list
        カット点のフレーム番号リスト

    diff_images : numpy.ndarray
        隣接フレーム間の差分画像

    Returns
    -------
    cut_point : list
        カット点（のフレーム番号リスト）
    """
    diff = np.diff(np.array(cut_point)).tolist() # 前後フレームのフレーム番号で差異を取る（操作のために一度型変換）
    diff_index = [i for i, diff in enumerate(diff) if diff == 1]    # 差異が1フレーム（連続なカット点）の添え字を取得
    remove_candidate = [cut_point[i+1] for i in diff_index]    # 削除候補
    
    delete_target = []  # 削除対象
    for i in remove_candidate:
        # 差分画像同士のMSEが閾値以下のとき削除対象
        if MSE(diff_images[i-1] - diff_images[i]) <= CUT_BETWEEN_THRESHOLD:
            delete_target.append(i)

    cut_point = target_delete(cut_point, delete_target) # 削除対象の全フレームを削除
    
    return cut_point

def create_mask_img(img):
    """
    マスク画像を作成する関数
    輝度ヒストグラムを求める際に使用

    Parameters
    ----------
    img : numpy.ndarray
        マスク画像を重ねたい画像

    Returns
    -------
    mask : numpy.ndarray
        カット検出点（フレーム番号）のリスト 
    """
    width, height = img.shape[:2]   # 幅、高さ
    
    w, h = img.shape[:2]
    mask = np.zeros((width, height), np.uint8)  # 初期化
    mask[w//5:w//5*4, h//5:h//5*4] = 255    # 中央だけ通すマスク画像

    return mask

def color_histogram_comp(img1, img2, THRESHOLD, need_mask=False):
    """
    2つの画像から輝度ヒストグラムの類似度を算出する関数
    類似度が3つ以上閾値を超えたとき削除対象として比較結果を返す

    Parameters
    ----------
    img1 : numpy.ndarray
        比較画像１の画像データ

    img2 : numpy.ndarray
        比較画像２の画像データ
    
    THRESHOLD : float
        使用する閾値

    need_mask : bool, default False
        マスク画像を適用するかどうか

    Returns
    -------
    bool
        比較結果（削除対象かどうか）
    """
    # 画像データからRGBの3色を取り出す
    b1, g1, r1 = img1[:,:,0], img1[:,:,1], img1[:,:,2]
    b2, g2, r2 = img2[:,:,0], img2[:,:,1], img2[:,:,2]

    # マスク画像を適用する場合
    if need_mask:
        mask = create_mask_img(img1)    # 中央だけを通すマスク画像を作成
        # 比較画像１のRed, Green, Blue、Gray の4つのヒストグラム作成する
        hist1_r = cv2.calcHist([r1],[0],mask,[256],[0,256])
        hist1_g = cv2.calcHist([g1],[0],mask,[256],[0,256])
        hist1_b = cv2.calcHist([b1],[0],mask,[256],[0,256])
        hist1_k = cv2.calcHist([img1],[0],mask,[256],[0,256])
        # 比較画像２も同様に作成
        hist2_r = cv2.calcHist([r2],[0],mask,[256],[0,256])
        hist2_g = cv2.calcHist([g2],[0],mask,[256],[0,256])
        hist2_b = cv2.calcHist([b2],[0],mask,[256],[0,256])
        hist2_k = cv2.calcHist([img2],[0],mask,[256],[0,256])
    else:
        hist1_r = cv2.calcHist([r1],[0],None,[256],[0,256])
        hist1_g = cv2.calcHist([g1],[0],None,[256],[0,256])
        hist1_b = cv2.calcHist([b1],[0],None,[256],[0,256])
        hist1_k = cv2.calcHist([img1],[0],None,[256],[0,256])

        hist2_r = cv2.calcHist([r2],[0],None,[256],[0,256])
        hist2_g = cv2.calcHist([g2],[0],None,[256],[0,256])
        hist2_b = cv2.calcHist([b2],[0],None,[256],[0,256])
        hist2_k = cv2.calcHist([img2],[0],None,[256],[0,256])

    # ヒストグラムインタセクション で比較して類似度を出す
    similarity_hist_r = cv2.compareHist(hist1_r, hist2_r, cv2.HISTCMP_INTERSECT) / np.sum(hist1_r)
    similarity_hist_g = cv2.compareHist(hist1_g, hist2_g, cv2.HISTCMP_INTERSECT) / np.sum(hist1_g) 
    similarity_hist_b = cv2.compareHist(hist1_b, hist2_b, cv2.HISTCMP_INTERSECT) / np.sum(hist1_b)
    similarity_hist_k = cv2.compareHist(hist1_k, hist2_k, cv2.HISTCMP_INTERSECT) / np.sum(hist1_k) 

    exceed_cnt = 0  # 閾値を超えた個数

    if similarity_hist_r >= THRESHOLD:
        exceed_cnt += 1
    if similarity_hist_g >= THRESHOLD:
        exceed_cnt += 1
    if similarity_hist_b >= THRESHOLD:
        exceed_cnt += 1
    if similarity_hist_k >= THRESHOLD:
        exceed_cnt += 1

    # 閾値を超える個数が3以上の時、削除対象とする
    if exceed_cnt >= 3:
        return True
    else:
        return False
        
def incorrect_cut_point_delete_by_color_histogram(cut_point, frames):
    """
    輝度ヒストグラム（カラーヒストグラム）による誤ったカット点の削除を行う関数

    [削除の理由]
        動きの激しい時に大量かつ連続でカット点を取ってしまう問題が生じているため
    [方法]
        カット点の画像データから Red, Green, Blue、Gray の4つのヒストグラムを作成する
        ヒストグラムの作成時には、中央のみを通すマスク画像を使用する（中央部分のみの輝度ヒストグラム）
        1つ後ろのカット点に対しても同様に4つのヒストグラムを作成して2つのヒストグラムを比較する
        比較には、ヒストグラムインタセクション を使って、類似度を出す
        類似度（高いほど類似している）が閾値以上の時（※）カット点から削除
        ※RGBとGray の4つのヒストグラムの類似度が3つ以上閾値を超える時

        カット点同士の類似度が高い → 誤ったカット点の可能性が高い（カット点に同じような画像は基本的にないため）

        ヒストグラムインタセクション    D = Σ min(h[i] - h[i+1])　　h はヒストグラム
        
    具体的な処理のコードはサブ関数である[color_histogram_comp]に記載

    Parameters
    ----------
    cut_point : list
        カット点（のフレーム番号リスト）

    frames : numpy.ndarray
        フレームデータ（動画の全画像データ）

    Returns
    -------
    cut_point : list
        カット間フレームを削除した後のカット点
    """
    cut_frame = [frames[i] for i in cut_point]  # カット点の画像データ（フレーム）
    delete_target = []  # 削除対象

    for i in range(len(cut_frame)-1):
        isdelete = color_histogram_comp(cut_frame[i], cut_frame[i+1], HIST_THRESHOLD, need_mask=True)   # 2つの画像の比較結果（削除対象かどうか）

        # 削除対象の時
        if isdelete:
            delete_target.append(cut_point[i])

    cut_point = target_delete(cut_point, delete_target) # 削除対象の全フレームを削除
    
    return cut_point

def flash_frame_delete(cut_point, frames):
    """
    フラッシュを検出して、該当カット点を削除する関数
    [方法]
        次のカット点とのフレーム差が5フレーム以内のカット点のみ検査する
        次のカット点を含めて幅3フレーム分の画像を取得する　
        取得した複数の画像の各画素の最小値を取った画像を作成する
        該当カット点の画像と作成した最小画素の画像で輝度ヒストグラムの類似度を算出
        類似度が閾値以上の時、フラッシュの可能性があるのでカット点から削除
        
        フラッシュは中央以外の場所でも起きるため、ヒストグラム作成時のマスク画像は適用しない

    Parameters
    ----------
    cut_point : list
        カット点（のフレーム番号リスト）

    frames : numpy.ndarray
        フレームデータ（動画の全画像データ）

    Returns
    -------
    cut_point : list
        カット間フレームを削除した後のカット点
    """
    delete_target = []  # 削除対象
    for i in range(len(cut_point)-1):
        # 次のカット点とのフレーム差が5フレーム以内の時
        if abs(cut_point[i] - cut_point[i+1]) <= 5: 
            # 3フレーム分の画像を取得
            range_images = []   
            for at in range(3):
                # 最後のフレーム番号を超える場合は、最後のフレームにする
                if cut_point[i+1] + at >= cut_point[-1]:
                    range_images.append(frames[cut_point[-1]])         
                else:         
                    range_images.append(frames[cut_point[i+1] + at])  
            # 比較画像の作成
            prev_frame = frames[cut_point[i]]        # 比較対象1
            next_frame = np.min(range_images, axis=0) # 比較対象2

            isdelete = color_histogram_comp(prev_frame, next_frame, FLASH_THRESHOLD, need_mask=False)   # 2つの画像の比較結果（削除対象かどうか）

            # 削除対象の時
            if isdelete:
                delete_target.append(cut_point[i])
                delete_target.append(cut_point[i+1])

    cut_point = target_delete(cut_point, delete_target) # 削除対象の全フレームを削除
    
    return cut_point

def effect_frame_delete(cut_point, frames):
    """
    エフェクトを検出して、該当カット点を削除する関数
    [方法]
        次のカット点とのフレーム差が5フレーム以内のカット点のみ検査する
        次のカット点から幅5フレーム分の画像を取得する
        該当カット点の画像と取得した5つの画像で輝度ヒストグラムの類似度を算出（5回分）
        類似度が閾値以上の時、エフェクトの可能性があるのでカット点から削除
    
    Parameters
    ----------
    cut_point : list
        カット点（のフレーム番号リスト）

    frames : numpy.ndarray
        フレームデータ（動画の全画像データ）

    Returns
    -------
    cut_point : list
        カット間フレームを削除した後のカット点
    """
    delete_target = []
    for i in range(len(cut_point)-1):
        if abs(cut_point[i] - cut_point[i+1]) <= 5: 
            for at in range(1, 6):  # 5フレーム分
                prev_frame = frames[cut_point[i]]   # 比較対象1

                # 最後のフレーム番号を超える場合は、最後のフレームにする
                if cut_point[i+1] + at >= cut_point[-1]:
                    next_frame = frames[cut_point[-1]]          # 比較対象2
                else:            
                    next_frame = frames[cut_point[i+1] + at]    # 比較対象2

                isdelete = color_histogram_comp(prev_frame, next_frame, EFFECT_THRESHOLD, need_mask=False)   # 2つの画像の比較結果（削除対象かどうか）

                # 削除対象の時
                if isdelete:
                    delete_target.append(cut_point[i])
                    delete_target.append(cut_point[i+1])

    delete_target = list(set(delete_target))    # 重複を削除
    cut_point = target_delete(cut_point, delete_target) # 削除対象の全フレームを削除
                
    return cut_point

def read_video_data(input_video_path):
    """
    動画を読み込み、フレームデータと動画情報を抽出する関数

    Parameters
    ----------
    input_video_path : str
        動画の入力パス   

    Returns
    -------
    frames : numpy.ndarray
        フレームデータ（動画の全画像データ）
    
    video_info : list 
        動画データ [fps, width, height]
    """
    # --------------------------------------------------
    # 動画の読み込み
    # --------------------------------------------------
    cap = cv2.VideoCapture(input_video_path)     
    # ビデオキャプチャーが開けていない場合、例外を返す
    if cap.isOpened() is False:
        raise ValueError('読み込みエラー : 動画ID ' + video_id + 'が上手く読み取れません。')

    fps = cap.get(cv2.CAP_PROP_FPS)                 # FPS
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)       # 幅
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)     # 高さ
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # 総フレーム数
    sec = frame_count / fps                         # 秒数
    
    # --------------------------------------------------
    # フレーム毎の画像情報をリストに格納
    # --------------------------------------------------
    n_frames = int(frame_count) # 総フレーム数 
    frames = []
    while True:
        ret, frame = cap.read()
        # 最後まで取得出来なかった場合、そこまでを総フレームとして更新
        if not ret:
            #n_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            break

        frames.append(frame)
        if len(frames) == n_frames:
            break

    if cap.isOpened():
        cap.release()

    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]   # 処理のため、BGRからRGBにする
    video_info = [fps, width, height]   # 戻り値用の動画情報をまとめる

    return frames, video_info

def cut_point_detect(frames):
    """
    カット点を検出して、返す関数
    [手順]
    1. 隣接フレーム間で差分画像を作成
    2. 変化割合を算出
    3. 変化割合が閾値以上の時、カット点として抽出
    4. 段階的なカット点の修正
        4-1 カット間フレーム（不要フレーム）を削除
        4-2 輝度ヒストグラムで誤ったカット点を削除
        4-3 フラッシュ検出による誤ったカット点を削除
        4-4 エフェクト検出による誤ったカット点を削除

    Parameters
    ----------
    frames : numpy.ndarray
        フレームデータ（動画の全画像データ） 

    Returns
    -------
    cut_point : list
        カット検出点（フレーム番号）のリスト 

    """
    # --------------------------------------------------
    # 1. 隣接フレーム間で差分画像を作成
    # --------------------------------------------------
    diff_images = [aft - bef for bef, aft in zip(frames, frames[1:])]   # 差分画像

    # --------------------------------------------------
    # 2. 変化割合を算出
    # --------------------------------------------------
    diff_rates = [MSE(d) for d in diff_images]  # 差分画像のMSEを変化割合とする
    
    # --------------------------------------------------
    # 3. 変化割合が閾値以上の時、カット点として抽出する 
    # --------------------------------------------------
    cut_point = []    # カット点のフレーム番号リスト
    for i in range(len(diff_images)):
        # 変化割合が閾値を以上の時
        if diff_rates[i] >= CUT_THRESHOLD:    
            cut_point.append(i)   # リストに追加

    # --------------------------------------------------
    # 4. 段階的なカット点の修正
    #--------------------------------------------------
    # 4-1 カット間フレーム（不要フレーム）を削除
    cut_point = cut_between_frame_delete(cut_point, diff_images)

    # 4-2 輝度ヒストグラムの類似度による誤ったカット点を削除
    cut_point = incorrect_cut_point_delete_by_color_histogram(cut_point, frames) 

    # 4-3 フラッシュ検出による誤ったカット点を削除
    cut_point = flash_frame_delete(cut_point, frames)

    # 4-4 エフェクト検出による誤ったカット点を削除
    cut_point = effect_frame_delete(cut_point, frames)
    
    cut_point.append(len(frames) - 1) # 動画の最後のフレームインデックスを追加

    return cut_point
  
def graph_save(data, dest_path):
    """
    変化割合のグラフを保存する関数

    Parameters
    ----------
    data : numpy.ndarray
        変化割合

    dest_path : str
        保存先フォルダのパス
    """
    fig = plt.figure(figsize=(10, 5))
    plt.plot(data)                           
    plt.title('Graph of Cut Change Detection')      # タイトル
    plt.xlabel('frame')                             # X軸ラベル
    plt.ylabel('Diff Rate')                         # Y軸ラベル
    plt.grid(True)                                  # グリッドあり
    plt.axhline(85, color='red')           # カットするラインを描画
    #plt.show() # グラフ描画
    
    plt.savefig(dest_path + '\\change_rate_graph.jpg') # 保存

def cut_save(video_id, cut_point, frames, video_info, dest_path):
    """
    動画を分割して保存する関数

    Parameters
    ----------
    video_id : str
        動画ID 

    cut_point : list
        カット検出点（フレーム番号）のリスト
    
    frames : numpy.ndarray
        フレームデータ（動画の全画像データ） 
    
    video_info : list 
        動画データ [fps, width, height]
    
    dest_path : str
        保存先フォルダのパス
    """
    # --------------------------------------------------
    # カット分割・保存
    # --------------------------------------------------
    fps, width, height = video_info # 動画情報の展開

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')    # 動画の保存形式
    writer = [] # カット書き込み用のリスト
    begin = 0   # カット最初のフレーム

    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]   # RGBからBGRに戻す（戻さないと色が反転したまま保存される）

    cut_count = len(cut_point) # カット数
    for i in range(cut_count):
        writer.append(cv2.VideoWriter(dest_path + '\\cut' + str(i+1) + '.mp4', fourcc, fps, (int(width), int(height))))
        for j in range(begin, cut_point[i]+1):
            writer[i].write(frames[j])
        begin = cut_point[i]+1

    logger.debug(video_id + '_cut1 ～ ' + str(cut_count) + 'を保存しました')
    logger.debug('保存先 : ' + dest_path)
    logger.debug('--------------------------------------------------')

    # --------------------------------------------------
    # 変化割合グラフを保存
    # --------------------------------------------------
    # diff_images = [aft - bef for bef, aft in zip(frames, frames[1:])]   # 差分画像
    # diff_rates = [MSE(d) for d in diff_images]  # 差分画像のMSEを変化割合とする
    # graph_save(frames, dest_path)

def cut_segmentation(video_path, result_cut_path):
    """
    カット分割を行い、各カットをフォルダに保存する関数
    
    以下の流れで行う
    
    動画IDリストの作成
    カット分割
        保存先フォルダの作成 
        動画の読み込み、フレームデータ，動画情報の抽出
        カット点の検出
        カットの保存

    Parameters
    ----------
    video_path : str
        入力する動画データのフォルダパス

    result_cut_path
        カット分割結果を保存するフォルダパス
    """
    # --------------------------------------------------
    # 動画IDリストの作成
    # --------------------------------------------------
    files = os.listdir(video_path)  # 動画ファイル名（動画ID）一覧を取得
    video_id_list = [f.replace('.mp4', '') for f in files]  # 一覧から拡張子を削除

    # --------------------------------------------------
    # カット分割
    # --------------------------------------------------
    for video_id in video_id_list:
        input_video_path = video_path + '\\' + video_id + '.mp4' # 動画ファイルの入力パス 
        
        # --------------------------------------------------
        # 保存先フォルダの作成
        # --------------------------------------------------
        dest_path = os.path.join(result_cut_path, video_id) # 各動画のカット分割結果の保存先
        dest_folder_create(dest_path)   # フォルダ作成 
        
        # --------------------------------------------------
        # 動画の読み込み、フレームデータと動画情報を抽出
        # --------------------------------------------------
        frames, video_info = read_video_data(input_video_path)
        
        # --------------------------------------------------
        # カット点の検出
        # --------------------------------------------------
        cut_point = cut_point_detect(frames)
        print(video_id, cut_point)
        
        # --------------------------------------------------
        # カット点の情報を使用して、動画を分割して保存
        # --------------------------------------------------
        cut_save(video_id, cut_point, frames, video_info, dest_path)