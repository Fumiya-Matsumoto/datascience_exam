# 求人原稿の応募数予測
  
## 概要
求人原稿に関するデータからその応募数を予測する機械学習モデルの予測結果を出力するWebアプリをFlaskで構築し、Herokuに公開する。

## 使い方
1. [Webページ](http://predict-num-apply.herokuapp.com/)にアクセスする。
2. 「ファイルを選択」をクリックして、テストデータを選択する。
3. 「アップロード」ボタンをクリックして、ファイルをアップロードする。
4. 推論が終わるまでしばらく待つ。
5. 推論が終わると、「ダウンロード」ボタンが表示されるので、ボタンをクリックしcsv形式の出力結果をダウンロードする。


<img src="./images/app_image_01.png" width="50%"><img src="./images/app_image_02.png" width="50%">

 
## ライブラリ
* Flask 1.1.2
* lightgbm 3.1.1
* numpy 1.21.2
* pandas 1.3.5
* scikit-learn 1.0.2
* xgboost 1.5.0
* catboost 0.26.1

## 起動方法
```bash
git clone https://github.com/Fumiya-Matsumoto/datascience_exam.git
cd datascience_exam
docker-compose up --build
```
※ Githubには学習データを上げてないので、このリポジトリをクローンして実行しても、アプリは動きません。`datascience_exam/app/input`の中に学習データを`train_x.csv`という名前で保存する必要があります。

## Herokuに公開する方法
まずは、Herokuへログインする。
```bash
heroku login
```
上記コマンドを実行すると、Herokuのログイン画面がブラウザ上に開かれるため、ログインする。

続いて、Heroku上にアプリケーションを作成する
```bash
heroku create APP_NAME
```
Heroku上のGitにpushすることがデプロイのトリガーとなるため、リモートリポジトリに追加する。
```bash
heroku git:remote -a APP_NAME
```
コンテナを用いることをスタックに指定しておく。
```bash
heroku stack:set container
```
Heroku上にPushする。
```bash
git push heroku master
```
これでデプロイが完了する。
