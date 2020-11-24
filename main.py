"""
PD3で実装したメインのプログラム

主な工程は以下の通り
・カット分割
・ラベル付け
・シーン統合
・分析
"""
import os
import os.path
from logging import getLogger, StreamHandler, DEBUG

from setting import path_setting
from file_io import read_csv
from cut_segmentation_mod import cut_segmentation
from cut_img_generate_mod import cut_img_generate

if __name__ == '__main__':
    # --------------------------------------------------
    # 処理開始
    # --------------------------------------------------
    try:   
        # --------------------------------------------------
        # 各種設定
        # --------------------------------------------------
        os.chdir(r'C:\Users\hukuyori\CM_Analysis')  # TODO 後で消す
        
        # path の設定
        path = path_setting()   
        root_path, video_path, cmData_path, result_cut_path, result_cut_img_path, ansData_path, result_eva_path = path 

        # ルートディレクトリではないとき作業ディレクトリを変更
        if os.getcwd() == root_path:
            os.chdir(root_path)

        # ログ設定
        logger = getLogger(__name__)
        handler = StreamHandler()
        handler.setLevel(DEBUG)
        logger.setLevel(DEBUG)
        logger.addHandler(handler)
        logger.propagate = False
            
        # --------------------------------------------------
        # CMデータの読み込み
        # --------------------------------------------------
        # ファイルの存在チェック
        if not os.path.exists(cmData_path):
            raise ValueError('CSVファイルを読み込めません。')

        cm_data = read_csv(cmData_path) # CMデータ（動画ID, 企業名, 商品名, 作品名, クラスタパターン）
        video_id_list = [cm_data[i][1] for i in range(len(cm_data))]    # 動画IDリストを作成

        logger.debug('CMデータを読み込みました。')
        logger.debug('入力元 : ' + cmData_path)
        logger.debug('--------------------------------------------------')

        # --------------------------------------------------
        # カット分割
        # --------------------------------------------------
        logger.debug('カット分割を開始します。')

        #cut_segmentation(video_path, result_cut_path)   # カット分割
        
        logger.debug('全動画のカット分割が終了しました。')
        logger.debug('--------------------------------------------------')

        # --------------------------------------------------
        # カット画像の作成
        # --------------------------------------------------
        logger.debug('カット画像生成を開始します。')

        cut_img_generate(video_path, result_cut_img_path)

        logger.debug('全カットのカット画像生成が終了しました。')
        logger.debug('--------------------------------------------------')

        # --------------------------------------------------
        # 物体認識によるラベル付け
        # --------------------------------------------------

        # --------------------------------------------------
        # 動作認識によるラベル付け
        # --------------------------------------------------

        # --------------------------------------------------
        # シーン統合
        # --------------------------------------------------

        # --------------------------------------------------
        # 分析
        # --------------------------------------------------

    # 例外処理
    except ValueError as e:
        print(e)
    except:
        import sys
        print("Error: ", sys.exc_info()[0])
        print(sys.exc_info()[1])
        import traceback
        print(traceback.format_tb(sys.exc_info[2]))


