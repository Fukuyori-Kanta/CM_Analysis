"""
研究で実装したメインの分析プログラム

主な工程は以下の通り
・カット分割
・ラベル付け
・シーン統合
・分析
"""
import os
import subprocess

from utils.init_setting import Path, setup_logger, get_env_data
from utils.file_io import read_csv
from utils.cut_segmentation_mod import cut_segmentation
from utils.cut_img_generate_mod import cut_img_generate
from utils.label_shaping_mod import label_shaping
from utils.scene_integration_mod import scene_integration
from utils.analysis_mod import favo_analysis

if __name__ == '__main__':
    # --------------------------------------------------
    # 処理開始
    # --------------------------------------------------
    try:   
        # --------------------------------------------------
        # 各種設定
        # --------------------------------------------------
        # スクリプト実行ディレクトリ（[main.py] のディレクトリ）に変更
        os.chdir(os.path.dirname(os.path.abspath(__file__)))  

        # ログ設定
        logger = setup_logger(__name__)

        # パス設定
        path = Path()

        # ルートディレクトリではないとき作業ディレクトリを変更
        if os.getcwd() == path.root_path:
            os.chdir(path.root_path)
        
        logger.debug('各種設定が完了しました。')
        
        # --------------------------------------------------
        # CMデータの読み込み
        # --------------------------------------------------
        # ファイルの存在チェック
        if not os.path.exists(path.cmData_top_path):
            raise ValueError('CMデータ(上位)を読み込めません。')
        if not os.path.exists(path.cmData_btm_path):
            raise ValueError('CMデータ(下位)を読み込めません。')

        # CMデータの取得（以下のデータ全て）
        """
        ['期間区分', '企業名', '商品名', '作品名', 'SEC', '好感度票数', '好感度Ｐ‰', '放送回数', 'GRP換算値', '出演者・キャラクター', 'グループ名', 
         'CMの主な情景・シーン', '音声・セリフ', '画面上の全コピー', '出演者・キャラクター', 'ユーモラス', 'セクシー', '宣伝文句', '音楽・サウンド', 
         '商品にひかれた', '説得力に共感した', 'ダサイけど憎めない', '時代の先端を感じた', '心がなごむ', 'ストーリー展開', '企業姿勢にウソがない', 
         '映像・画像', '周囲の評判も良い', 'かわいらしい', '男性全体', '女性全体', '男性6-12', '男性13-19', '男性20-29', '男性30-39', '男性40-49', 
         '男性50-59', '男性60以上', '女性6-12', '女性13-17', '独身女性18-24', '独身女性25-59', '主婦39以下', '主婦40-49', '主婦50-59', '女性60以上', 
         'SPONSOR_ID', 'GOODS_ID', 'WORK_ID', '映像コード', 'CM好感度sec1', 'CM好感度sec2', 'CM好感度sec3', 'CM好感度sec4', 'CM好感度sec5', 
         'CM好感度sec6', 'CM好感度sec7', 'CM好感度sec8', 'CM好感度sec9', 'CM好感度sec10', 'CM好感度sec11', 'CM好感度sec12', 'CM好感度sec13', 
         'CM好感度sec14', 'CM好感度sec15', 'CM好感度sec16', 'CM好感度sec17', 'CM好感度sec18', 'CM好感度sec19', 'CM好感度sec20', 'CM好感度sec21', 
         'CM好感度sec22', 'CM好感度sec23', 'CM好感度sec24', 'CM好感度sec25', 'CM好感度sec26', 'CM好感度sec27', 'CM好感度sec28', 'CM好感度sec29', 'CM好感度sec30']
        """
        # 上位データ
        header, *data_row = read_csv(path.cmData_top_path)
        cm_data_top = [dict(zip(header, item)) for item in data_row]    # 辞書型のリストに整形
        
        # 下位データ
        header, *data_row = read_csv(path.cmData_btm_path)
        cm_data_btm = [dict(zip(header, item)) for item in data_row]
        
        # 動画IDリストを作成
        video_id_list_top = [cm_data_top[i]['映像コード'] for i in range(len(cm_data_top))]  # 上位       
        video_id_list_btm = [cm_data_btm[i]['映像コード'] for i in range(len(cm_data_btm))]  # 下位
        video_id_list = video_id_list_top + video_id_list_btm                               # 全て

        logger.debug('CMデータを読み込みました。')
        logger.debug('入力元 : ' + path.cmData_top_path)
        logger.debug('入力元 : ' + path.cmData_btm_path)
        logger.debug('-' * 90)

        # --------------------------------------------------
        # カット分割
        # --------------------------------------------------
        logger.debug('カット分割を開始します。')

        cut_segmentation(video_id_list, path.video_dir, path.cut_dir, path.cut_point_path)   # カット分割
        
        logger.debug('全動画のカット分割が終了しました。')
        logger.debug('-' * 90)
        
        # --------------------------------------------------
        # カット画像の作成
        # --------------------------------------------------
        logger.debug('カット画像生成を開始します。')

        cut_img_generate(path.video_dir, path.cut_img_dir, path.cut_point_path)

        logger.debug('全カットのカット画像生成が終了しました。')
        logger.debug('-' * 90)
        
        # --------------------------------------------------
        # 物体検出によるラベル付け
        # --------------------------------------------------
        logger.debug('物体検出によるラベル付けを開始します。')
        
        # 環境データの取得
        object_detection_env, config_file, checkpoint_file, classes_file  = get_env_data('OBJECT_DET_ENV')

        # 物体検出
        cmd = f'conda run -n {object_detection_env} python utils/object_detection_mod.py '\
            f'{config_file} {checkpoint_file} {classes_file} {path.cut_img_dir} {path.noun_label_path}'
        subprocess.call(cmd, shell=True)
        
        logger.debug('物体検出によるラベル付けが終了しました。')
        logger.debug('-' * 90)

        # --------------------------------------------------
        # 動作認識によるラベル付け
        # --------------------------------------------------
        logger.debug('動作認識によるラベル付けを開始します。')
        
        # 環境データの取得
        action_recognition_env, config_file, checkpoint_file, classes_file  = get_env_data("ACTION_REC_ENV")

        # 動作認識
        cmd = f'conda run -n {action_recognition_env} python utils/action_recognition_mod.py '\
            f'{config_file} {checkpoint_file} {classes_file} {path.cut_dir} {path.verb_label_path}'
        subprocess.call(cmd, shell=True)
        
        logger.debug('動作認識によるラベル付けが終了しました。')
        logger.debug('-' * 90)
        
        # --------------------------------------------------
        # ラベルデータの整形（翻訳，スクリーニング，結合）
        # --------------------------------------------------      
        logger.debug('ラベルデータの整形を開始します。')

        label_shaping(path.noun_label_path, path.verb_label_path, path.label_path)

        logger.debug('ラベルデータの整形が終了しました。')
        logger.debug('-' * 90)
        
        # --------------------------------------------------
        # シーン統合
        # --------------------------------------------------
        logger.debug('シーンの統合・保存を開始します。')

        scene_integration(path.cut_point_path, path.label_path, path.video_dir, path.scene_dir, path.scene_data_path)

        logger.debug('シーンの統合・保存が終了しました。')
        logger.debug('-' * 90)

        # --------------------------------------------------
        # 好感度とのマッチング・分析
        # --------------------------------------------------
        logger.debug('好感度とのマッチング・分析を開始します。')

        favo_analysis(path.cmData_top_path, path.cmData_btm_path, path.scene_data_path, path.favo_dir)

        logger.debug('好感度とのマッチング・分析が終了しました。')
        logger.debug('-' * 90)
        
    # 例外処理
    except ValueError as e:
        print(e)
    except:
        import sys
        print('Error: ', sys.exc_info()[0])
        print(sys.exc_info()[1])
        import traceback
        print(traceback.format_tb(sys.exc_info[2]))