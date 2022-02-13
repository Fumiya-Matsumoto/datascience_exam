import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost

app = Flask(__name__)
port = int(os.environ.get('PORT', 5000))

def predict(filename):
    X_test = pd.read_csv(f'./static/input/{filename}')
    X_train = pd.read_csv('./input/train_x.csv')
    all_data = pd.concat([X_train, X_test], sort=False, ignore_index=True)

    # 削除カラム
    delete_columns_1 = [
        '（派遣先）概要　勤務先名（フリガナ）', '未使用.18', 'WEB関連のスキルを活かす', '未使用.6', '給与　経験者給与下限',
        '学生歓迎', '固定残業制 残業代 上限', '未使用.19', '主婦(ママ)・主夫歓迎', '給与　経験者給与上限', '人材紹介',
        'エルダー（50〜）活躍中', '未使用.12', '未使用.15', 'WEB面接OK', 'プログラム関連のスキルを活かす',
        '固定残業制 残業代に充当する労働時間数 上限', '未使用.21', 'Wワーク・副業可能', 'ネットワーク関連のスキルを活かす',
        '待遇・福利厚生', 'シニア（60〜）歓迎', 'ベンチャー企業', '少人数の職場', '寮・社宅あり', '17時以降出社OK',
        'ブロックコード1', 'フリー項目　タイトル', '未使用.2', '未使用.1', '日払い', '未使用.22', '未使用.9',
        '未使用.5', '未使用.17', '外国人活躍中・留学生歓迎', '勤務地　周辺情報', '応募先　最寄駅（駅名）',
        '仕事写真（下）　写真2　ファイル名', 'バイク・自転車通勤OK', '応募先　所在地　都道府県', '応募先　所在地　ブロックコード',
        'オープニングスタッフ', '仕事写真（下）　写真3　ファイル名', '未使用.16', '仕事写真（下）　写真2　コメント',
        '仕事写真（下）　写真1　ファイル名', '応募先　最寄駅（沿線名）', '勤務地　最寄駅3（駅からの交通手段）', '未使用.3',
        '募集形態', '未使用.20', 'ブランクOK', '仕事写真（下）　写真3　コメント', '未使用', '未使用.13',
        'フリー項目　内容', '勤務地　最寄駅3（分）', '無期雇用派遣', '応募先　備考', '未使用.10', '応募先　所在地　市区町村',
        '先輩からのメッセージ', '仕事写真（下）　写真1　コメント', '勤務地　最寄駅3（沿線名）', '（派遣以外）応募後の流れ',
        '（派遣先）概要　従業員数', '未使用.11', '電話応対なし', '週払い', '固定残業制 残業代 下限',
        '（派遣先）勤務先写真コメント', '未使用.14', '未使用.4', '固定残業制 残業代に充当する労働時間数 下限',
        'ブロックコード3', '勤務地　最寄駅3（駅名）', 'メモ', '未使用.7', 'これまでの採用者例', 'ブロックコード2',
        '応募先　名称', '応募拠点', '経験必須', '未使用.8',
        '動画コメント', '（派遣）応募後の流れ', '動画タイトル', '動画ファイル名',
        '勤務地固定', '週1日からOK', 'ミドル（40〜）活躍中', 'ルーティンワークがメイン',
        '対象者設定　年齢下限', '対象者設定　年齢下限', '給与/交通費　給与支払区分',
        'CAD関連のスキルを活かす', '固定残業制', '公開区分', '20代活躍中', '検索対象エリア',
        '就業形態区分', '30代活躍中', '雇用形態', 'Dip JobsリスティングS', '資格取得支援制度あり',
        '対象者設定　年齢上限', '社会保険制度あり', '残業月10時間未満', '履歴書不要', '研修制度あり',
        'DTP関連のスキルを活かす', '新卒・第二新卒歓迎', '産休育休取得事例あり', '対象者設定　性別',
        'WEB登録OK'
    ]
    all_data.drop(columns=delete_columns_1, axis=1, inplace=True)

    # 特徴量エンジニアリング

    # 勤務時間
    all_data['勤務開始時間'] = all_data['期間・時間　勤務時間'].str.extract('(.+?)〜', expand=False)
    all_data['勤務終了時間'] = all_data['期間・時間　勤務時間'].str.extract('〜(.+?)<', expand=False)
    # 残業時間
    all_data['残業時間_文字列'] = all_data[all_data['期間・時間　勤務時間'].str.contains('残業')]['期間・時間　勤務時間'].str.extract('<BR>(.+?)※', expand=True)
    all_data['残業時間_範囲'] = all_data['残業時間_文字列'].map({
        '※残業はほとんどありません。<BR>':0,
        '※残業はほとんどありません。繁忙期あり。<BR>':0,
        '※残業はほとんどありまん。<BR>':0,
        '※残業はほとんどありません！<BR>':0,
        '※残業はほとんどなし。<BR>':0,
        '※残業ほとんどありません。<BR>':0,
        '※残業はほとんどありません。繁忙期有。<BR>':0,
        '※残業はほとんどありません。繁忙時期有<BR>':0,
        '※残業はほとんどありません。。<BR>':0,
        '※残業は月１〜１５時間と少なめです。<BR>':8,
        '※残業は月１〜１５時間と少なめ。<BR>':8,
        '※残業は月１〜１５時間程度と少なめ。<BR>':8,
        '※残業は月５〜１５時間程度と少なめ。<BR>':10,
        '※残業は月５〜１５時間程度と少なめです。<BR>':10,
        '※残業は月５〜１５時間と少なめです。<BR>':10,
        '※残業は月５〜１５時間と少なめ。<BR>':10,
        '※残業は月５〜１５時間程と少なめです。。<BR>':10,
        '※残業は月５〜１５時間以下と少なめです。<BR>':10,
        '※残業は月１０時間程度と少なめ。<BR>':10,
        '※残業は月５〜１５時間と少なめ。繁忙期あり。<BR>':10,
        '※残業は月１〜２０時間程度と少なめです。<BR>':10.5,
        '※残業は月８〜１３時間と少なめです。<BR>':10.5,
        '※残業は月１〜２０時間程度と少なめ。<BR>':10.5,
        '※残業は月７〜１５時間と少なめ。<BR>':11,
        '※残業は月１０〜１３時間程度と少なめ。<BR>':11.5,
        '※残業は月１１〜１２時間程度と少なめ。<BR>':11.5,
        '※残業は月１０〜１５時間程度と少なめ。<BR>':12.5,
        '※残業は月１０〜１５時間と少なめです。<BR>':12.5,
        '※残業は月１０〜１５時間程度と少なめです。<BR>':12.5,
        '※残業は月５〜２０時間程度と少なめです。<BR>':12.5,
        '※残業は月１０〜１５時間と少なめ。<BR>':12.5,
        '※残業は月５〜２０時間程度と少なめ。<BR>':12.5,
        '※残業は月５〜２０時間と少なめです。<BR>':12.5,
        '※残業は月５〜２０時間と少なめ。<BR>':12.5,
        '※残業は月１０〜１５時間程度と少なめです<BR>':12.5,
        '※残業は月１０〜１５時間程度。<BR>':12.5,
        '※残業は月１０〜１６時間程度と少なめ。<BR>':13,
        '※残業は月１０〜１９時間程度と少なめ。<BR>':14.5,
        '※残業は月１０〜２０時間程度と少なめ。<BR>':15,
        '※残業は月１０〜２０時間と少なめです。<BR>':15,
        '※残業は月１０〜２０時間程度と少なめです。<BR>':15,
        '※残業は月１５時間程度と少なめです。<BR>':15,
        '※残業は月１５時間程度と少なめ。<BR>':15,
        '※残業は月１０〜２０時間と少なめ。<BR>':15,
        '※残業は月１５時間以下と少なめです。<BR>':15,
        '※残業は月１０〜２０時間程度と少なめです<BR>':15,
        '※残業は月１０〜２０時間と少なめ。繁忙期あり。<BR>':15,
        '※残業は月１５時間以下と少なめ。<BR>':15,
        '※残業は月１０〜２０時間程度。繁忙期あり。<BR>':15,
        '※残業は月１５時間と少なめです。<BR>':15,
        '※残業は月１５時間と少なめ。<BR>':15,
        '※残業月１０〜２０時間程度と少なめ。<BR>':15,
        '※残業は月１５〜１９時間程度と少なめ。<BR>':17,
        '※残業は月１５〜２０時間程度と少なめ。<BR>':17.5,
        '※残業は月１５〜２０時間と少なめです。<BR>':17.5,
        '※残業は月１５〜２０時間程度と少なめです。<BR>':17.5,
        '※残業は月１５〜２０時間と少なめ。<BR>':17.5,
        '※残業は月１５〜２０時間と少なめ<BR>':17.5,
        '※残業は月１５月〜２０時間程度と少なめ。<BR>':17.5,
        '※残業は月１９時間程度と少なめです。<BR>':19,
        '※残業は月１９時間程度と少なめ。<BR>':19,
        '※残業は月１８〜２０時間程度と少なめ。<BR>':19,
        '※残業は月２０時間程度と少なめです。<BR>':20,
        '※残業は月２０時間程度と少なめ。<BR>':20,
        '※残業は月２０時間以下と少なめです。<BR>':20,
        '※残業は月２０時間以下と少なめ。<BR>':20,
        '※残業は月２０時間と少なめ。<BR>':20,
        '※残業は月２０時間と少なめです。<BR>':20,
        '※残業は月２０時間程度と少なめ。繁忙期有。<BR>':20,
        '※残業は月２０時間程度と少なめです<BR>':20,
        '※残業は月２０時間程度と少なめ<BR>':20
    })

    # 勤務開始年月日
    all_data['勤務開始年'] = all_data['期間・時間　勤務開始日'].str.extract('(.+?)/', expand=False).astype(int)
    all_data['勤務開始月'] = all_data['期間・時間　勤務開始日'].str.extract('/(.+)/', expand=False).astype(int)

    # 年間休日
    all_data['（紹介予定）休日休暇'] = all_data['（紹介予定）休日休暇'].str.extract('年間休日(.+)日', expand=False)
    holydays = all_data[all_data['（紹介予定）休日休暇'].notnull()]['（紹介予定）休日休暇'].map(int)
    all_data['（紹介予定）休日休暇'] = holydays

    # 勤務先写真ファイル（ラベルエンコーディングするかも）
    all_data.loc[all_data['（派遣先）勤務先写真ファイル名'].notnull(), '（派遣先）勤務先写真ファイル名'] = 0
    all_data.loc[all_data['（派遣先）勤務先写真ファイル名'].isnull(), '（派遣先）勤務先写真ファイル名'] = 1
    all_data['（派遣先）勤務先写真ファイル名'] = all_data['（派遣先）勤務先写真ファイル名'].astype(int)

    # 入社時期
    all_data['（紹介予定）入社時期'] = all_data['（紹介予定）入社時期'].map({
        '◆6ヶ月後':6,
        '◆5ヶ月後':5,
        '◆4ヶ月後':4,
        '◆3ヶ月後':3,
        '◆2ヶ月後':2,
        '◆1ヶ月後':1,
        '※ご紹介先により異なります。詳細はお問い合わせ下さい。':np.nan
    })

    # 給与/交通費
    def replace_fee(x):
        if x[-1] == '万':
            return int(x.replace('万', '0000'))
        elif x[-2] == '万':
            return int(x.replace('万', '000'))
        elif x[-3] == '万':
            return int(x.replace('万', '00'))
        elif x[-4] == '万':
            return int(x.replace('万', '0'))
        else:
            return int(x.replace('万', ''))
    all_data['月収'] = all_data['給与/交通費　備考'].str.extract('【月収例】(.+?)円', expand=False)
    all_data['月収'] = all_data[all_data['月収'].notnull()]['月収'].map(replace_fee)

    # 給与上限
    all_data['給与/交通費　給与上限_階層'] = pd.qcut(all_data['給与/交通費　給与上限'], 5, duplicates='drop')

    # 雇用形態
    for i in all_data[(all_data['（紹介予定）雇用形態備考']=='契約契約社員') | (all_data['（紹介予定）雇用形態備考']=='契約員')].index:
        all_data[i:i+1]["（紹介予定）雇用形態備考"] = "契約社員"

    all_data['お仕事No._label'] = all_data['お仕事No.']
    job_No_train = all_data[:len(X_train)]['お仕事No.']
    job_No_test = all_data[len(X_train):]['お仕事No.']
    job_No_test.reset_index(inplace=True, drop=True)

    # 削除するカラム
    delete_columns_2 = [
        '掲載期間　開始日', '掲載期間　終了日', '休日休暇　備考', 'お仕事名',  '仕事内容',
        '応募資格', 'お仕事のポイント（仕事PR）', '残業時間_文字列', '期間・時間　勤務時間',
        '期間・時間　勤務開始日', '給与/交通費　備考', '（紹介予定）年収・給与例', 'お仕事No.',
        '（派遣先）配属先部署　男女比　女'
    ]

    all_data.drop(columns=delete_columns_2, axis=1, inplace=True)

    # ラベルエンコーディングが必要なカラム
    label_encoding_columns = [
        '勤務地　最寄駅1（駅名）', '勤務地　最寄駅1（沿線名）', '勤務地　最寄駅2（駅名）', '勤務地　最寄駅2（沿線名）',
        '勤務地　備考', '拠点番号', '（派遣先）概要　勤務先名（漢字）', '派遣会社のうれしい特典', '（紹介予定）雇用形態備考',
        '（紹介予定）雇用形態備考', '期間･時間　備考', '（派遣先）職場の雰囲気', '勤務開始時間', '勤務終了時間',
        '（派遣先）配属先部署', '（紹介予定）待遇・福利厚生', '期間･時間　備考', '（派遣先）概要　事業内容', 'お仕事No._label',
        '給与/交通費　給与上限_階層', 
    ]

    for column in label_encoding_columns:
        le = LabelEncoder()
        le = le.fit(all_data[column])
        all_data[column] = le.transform(all_data[column])

    X_test = all_data[len(X_train):]
    X_test.reset_index(inplace=True, drop=True)
    
    load_models = []
    y_preds = []

    for model_name in ['lgb', 'xgb', 'cat']:
        for i in range(5):
            filename = f'./models/{model_name}_{i}.pkl'
            load_model = pickle.load(open(filename, 'rb'))
            if model_name == 'lgb':
                y_pred = load_model.predict(X_test, num_iteration=load_model.best_iteration)
            elif model_name == 'xgb':
                xgb_test = xgb.DMatrix(X_test)
                y_pred = load_model.predict(xgb_test, iteration_range=(0, load_model.best_iteration))
            else:
                y_pred = load_model.predict(X_test)
            y_preds.append(y_pred)
    
    y_sub = sum(y_preds) / len(y_preds)

    sub = pd.DataFrame(
        {
        'お仕事No.': job_No_test,
        '応募数 合計': y_sub
        })

    sub.loc[sub['応募数 合計'] < 0, '応募数 合計'] = 0
    
    sub.to_csv(f'./static/output/predict.csv', index=False)


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')
    elif request.method == 'POST':
        file = request.files['file']
        save_filename = secure_filename(file.filename)
        file.save(os.path.join('./static/input', save_filename))
        predict(save_filename)
        return redirect(url_for('uploaded_file', filename=save_filename))

@app.route('/uploaded_file/<string:filename>')
def uploaded_file(filename):
    return render_template('uploaded_file.html', filename=filename)

@app.route('/download', methods=['GET'])
def download_api():
    filename = 'predict.csv'
    filepath = f'./static/output/{filename}'
    return send_file(
        filepath, as_attachment=True,
        attachment_filename=filename,
        mimetype='text/csv')

if __name__ == '__main__':
    app.run(
        debug=bool(os.environ.get('DEBUG', False)),
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)))