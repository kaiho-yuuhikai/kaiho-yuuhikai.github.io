#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geminiを使った感情分析と全体満足度の高精度補完
"""

import pandas as pd
import numpy as np
import json
import time
import os
from google import genai
from google.genai.types import GenerateContentConfig
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Vertex AI経由で接続（globalリージョン必須）
client = genai.Client(
    vertexai=True,
    project="gscale-george-test",  # プロジェクトID
    location="global"
)

def score_to_numeric(text):
    """満足度テキストを数値に変換"""
    if pd.isna(text):
        return np.nan
    try:
        val = float(text)
        if not np.isna(val):
            return val
    except:
        pass
    for i in range(5, 0, -1):
        if str(i) in str(text):
            return float(i)
    return np.nan

def analyze_satisfaction_with_gemini(row_data):
    """Geminiで全回答項目から満足度スコアを推定"""
    # 有効なデータがあるか確認
    has_data = False
    for key, val in row_data.items():
        if val and str(val).strip() and str(val) != 'nan':
            has_data = True
            break

    if not has_data:
        return None, None

    prompt = f"""あなたは同窓会イベントの参加者アンケートを分析するエキスパートです。

以下は「第3回 開邦高校大同窓会」（533名参加、沖縄県立開邦高校の同窓会）の事後アンケート回答です。
この回答者のイベント全体に対する満足度を1〜5のスケールで推定してください。

【評価の観点】
- 自由記述のトーン（ポジティブ/ネガティブ）
- 今後の協力意向の強さ（多く選択している＝エンゲージメントが高い＝満足度が高い傾向）
- 連絡先を提供しているか（実名・メール記入＝信頼・満足の表れ）
- メッセージの内容（感謝、要望、批判など）

【評価基準】
5: 非常に満足（強い感謝、積極的な協力意向、ポジティブなコメント）
4: 満足（概ね肯定的、協力意向あり、軽微な要望程度）
3: 普通（中立的、特に強い感情表現なし）
2: やや不満（改善要望が中心、ネガティブな表現が目立つ）
1: 不満（強い不満・批判）

【回答内容】
■ プログラム・演出についてのコメント:
{row_data.get('program_comment', '（未記入）')}

■ 飲食・会場・運営についてのコメント:
{row_data.get('venue_comment', '（未記入）')}

■ 今後の協力意向（選択項目）:
{row_data.get('cooperation', '（未選択）')}

■ 連絡可否:
{row_data.get('contact_ok', '（未回答）')}

■ 協力可能な分野・メッセージ:
{row_data.get('message', '（未記入）')}

【出力形式】
以下のJSON形式のみで回答してください（説明文は不要）：
{{"score": <1-5の数値>, "reasoning": "<30文字以内の根拠>"}}
"""

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1000,
            )
        )

        result_text = response.text
        if result_text is None:
            return None, None
        result_text = result_text.strip()
        # JSON部分を抽出
        if '{' in result_text and '}' in result_text:
            json_str = result_text[result_text.index('{'):result_text.rindex('}')+1]
            result = json.loads(json_str)
            return result.get('score'), result.get('reasoning')
    except Exception as e:
        print(f"Error: {e}")
        return None, None

    return None, None

def quantify_engagement(row, df_columns):
    """エンゲージメント指標を定量化"""
    engagement_score = 0

    # 連絡可否（名前・メール記入）
    name_col = [c for c in df_columns if 'お名前' in c]
    email_col = [c for c in df_columns if 'メール' in c]
    contact_col = [c for c in df_columns if '連絡を差し上げて' in c]

    if name_col and pd.notna(row.get(name_col[0])):
        engagement_score += 1  # 実名記入
    if email_col and pd.notna(row.get(email_col[0])):
        engagement_score += 1  # メール記入
    if contact_col and 'はい' in str(row.get(contact_col[0], '')):
        engagement_score += 1  # 連絡OK

    # 協力意向の数
    coop_col = [c for c in df_columns if '協力' in c and '形' in c]
    if coop_col:
        coop_text = str(row.get(coop_col[0], ''))
        if coop_text and coop_text != 'nan':
            coop_count = len([x for x in coop_text.split(',') if x.strip()])
            engagement_score += min(coop_count, 3)  # 最大3点

    # 今後参加したい活動の数
    activity_col = [c for c in df_columns if '今後' in c and '参加' in c]
    if activity_col:
        activity_text = str(row.get(activity_col[0], ''))
        if activity_text and activity_text != 'nan':
            activity_count = len([x for x in activity_text.split(',') if x.strip()])
            engagement_score += min(activity_count, 2)  # 最大2点

    return engagement_score

def main():
    import sys
    print("=" * 60, flush=True)
    print("Gemini感情分析による全体満足度の高精度補完", flush=True)
    print("=" * 60, flush=True)

    # データ読み込み
    df = pd.read_csv('【卒業生・教職員用】第3回 開邦高校大同窓会 事後アンケート（回答） - フォームの回答 1 (3).tsv',
                     sep='\t', encoding='utf-8')
    print(f"\n[1] データ読み込み: {len(df)}件")

    # 満足度を数値化
    satisfaction_cols = {
        '料理': '料理・ドリンク',
        '会費': '会費の妥当性',
        '日程': '開催日程',
        '時間帯': '開催時間帯',
        '全体': '全体満足度',
    }

    for name, keyword in satisfaction_cols.items():
        matched = [c for c in df.columns if keyword in c]
        if matched:
            df[f'{name}_数値'] = df[matched[0]].apply(score_to_numeric)

    print(f"  全体満足度: 非欠損 {df['全体_数値'].notna().sum()}件, 欠損 {df['全体_数値'].isna().sum()}件")

    # テキスト列を特定
    text_cols = [
        'プログラムや演出（セッション・演奏等）について',
        '飲食・会場環境・運営全般について',
    ]

    # 関連する列を特定
    program_col = [c for c in df.columns if 'プログラム' in c and '演出' in c]
    venue_col = [c for c in df.columns if '飲食' in c and '会場' in c]
    coop_col = [c for c in df.columns if '協力' in c and '形' in c]
    contact_col = [c for c in df.columns if '連絡を差し上げて' in c]
    message_col = [c for c in df.columns if 'メッセージ' in c]

    print(f"\n  検出した列:")
    print(f"    プログラム・演出: {program_col[0] if program_col else '未検出'}")
    print(f"    飲食・会場: {venue_col[0] if venue_col else '未検出'}")
    print(f"    協力意向: {coop_col[0] if coop_col else '未検出'}")
    print(f"    連絡可否: {contact_col[0] if contact_col else '未検出'}")
    print(f"    メッセージ: {message_col[0] if message_col else '未検出'}")

    # Geminiで感情分析
    print("\n[2] Gemini感情分析...")

    # 結果を保存するファイルがあれば読み込み
    cache_file = 'gemini_sentiment_cache.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            sentiment_cache = json.load(f)
        print(f"  キャッシュ読み込み: {len(sentiment_cache)}件")
    else:
        sentiment_cache = {}

    gemini_scores = []
    gemini_reasonings = []

    for idx in df.index:
        # 全フィールドを抽出
        row_data = {
            'program_comment': str(df.loc[idx, program_col[0]]) if program_col and pd.notna(df.loc[idx, program_col[0]]) else None,
            'venue_comment': str(df.loc[idx, venue_col[0]]) if venue_col and pd.notna(df.loc[idx, venue_col[0]]) else None,
            'cooperation': str(df.loc[idx, coop_col[0]]) if coop_col and pd.notna(df.loc[idx, coop_col[0]]) else None,
            'contact_ok': str(df.loc[idx, contact_col[0]]) if contact_col and pd.notna(df.loc[idx, contact_col[0]]) else None,
            'message': str(df.loc[idx, message_col[0]]) if message_col and pd.notna(df.loc[idx, message_col[0]]) else None,
        }

        # キャッシュチェック
        cache_key = str(idx)
        if cache_key in sentiment_cache:
            score = sentiment_cache[cache_key]['score']
            reasoning = sentiment_cache[cache_key].get('reasoning', '')
        else:
            print(f"  分析中: {idx+1}/{len(df)} ...", end=' ')
            score, reasoning = analyze_satisfaction_with_gemini(row_data)
            if score is not None:
                sentiment_cache[cache_key] = {'score': score, 'reasoning': reasoning}
                print(f"スコア: {score} ({reasoning})")
            else:
                print("スキップ（データなし）")
            time.sleep(0.5)  # レート制限対策

        gemini_scores.append(score)
        gemini_reasonings.append(reasoning if score else None)

    # キャッシュ保存
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(sentiment_cache, f, ensure_ascii=False, indent=2)

    df['gemini_sentiment'] = gemini_scores
    df['gemini_reasoning'] = gemini_reasonings
    valid_gemini = df['gemini_sentiment'].notna().sum()
    print(f"  Gemini分析完了: {valid_gemini}件")

    # エンゲージメントスコア計算
    print("\n[3] エンゲージメント指標の定量化...")
    df['engagement_score'] = df.apply(lambda row: quantify_engagement(row, df.columns), axis=1)
    print(f"  エンゲージメントスコア: 平均 {df['engagement_score'].mean():.2f}, 最大 {df['engagement_score'].max()}")

    # 相関分析
    print("\n[4] 特徴量と全体満足度の相関分析...")
    valid = df[df['全体_数値'].notna()].copy()

    features = ['料理_数値', '会費_数値', '日程_数値', '時間帯_数値', 'gemini_sentiment', 'engagement_score']
    print("  特徴量        | 相関係数")
    print("  " + "-" * 30)
    for f in features:
        if f in valid.columns and valid[f].notna().sum() > 10:
            corr = valid['全体_数値'].corr(valid[f])
            print(f"  {f:15s} | {corr:.3f}")

    # 回帰モデル構築
    print("\n[5] 回帰モデル構築...")

    # 学習データの準備
    feature_cols = ['料理_数値', '会費_数値', '日程_数値', '時間帯_数値']

    # Geminiセンチメントがあれば追加
    if valid['gemini_sentiment'].notna().sum() > 20:
        feature_cols.append('gemini_sentiment')

    # エンゲージメントスコアを追加
    feature_cols.append('engagement_score')

    # 欠損のないデータで学習
    train_mask = valid['全体_数値'].notna()
    for col in feature_cols:
        train_mask &= valid[col].notna()

    train_df = valid[train_mask].copy()
    print(f"  学習データ: {len(train_df)}件")

    X_train = train_df[feature_cols].values
    y_train = train_df['全体_数値'].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"  R²スコア: {model.score(X_train, y_train):.3f}")

    # 交差検証
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"  交差検証R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

    print("\n  係数:")
    for f, coef in zip(feature_cols, model.coef_):
        print(f"    {f}: {coef:.4f}")
    print(f"    切片: {model.intercept_:.4f}")

    # 欠損値の補完
    print("\n[6] 欠損値の補完...")
    df['全体_補完'] = df['全体_数値'].copy()

    missing_mask = df['全体_数値'].isna()
    impute_mask = missing_mask.copy()
    for col in feature_cols:
        if col != 'gemini_sentiment':  # Geminiセンチメントは欠損許容
            impute_mask &= df[col].notna()

    impute_df = df[impute_mask].copy()
    print(f"  補完対象: {len(impute_df)}件")

    if len(impute_df) > 0:
        X_impute = impute_df[feature_cols].fillna(impute_df[feature_cols].median()).values
        imputed_values = model.predict(X_impute)
        imputed_values = np.clip(imputed_values, 1, 5)

        df.loc[impute_mask, '全体_補完'] = imputed_values

        print(f"  補完値平均: {imputed_values.mean():.3f}")
        print(f"  補完値範囲: {imputed_values.min():.2f} - {imputed_values.max():.2f}")

    # 結果サマリー
    print("\n" + "=" * 60)
    print("結果サマリー")
    print("=" * 60)

    actual_mean = df[df['全体_数値'].notna()]['全体_数値'].mean()
    imputed_mean = df['全体_補完'].mean()

    print(f"  実測平均（50件）: {actual_mean:.3f}")
    print(f"  補完後平均（119件）: {imputed_mean:.3f}")

    # 4点以上の割合
    actual_4plus = (df[df['全体_数値'].notna()]['全体_数値'] >= 4).mean() * 100
    all_4plus = (df['全体_補完'] >= 4).mean() * 100
    print(f"  4点以上の割合（実測）: {actual_4plus:.1f}%")
    print(f"  4点以上の割合（全体）: {all_4plus:.1f}%")

    # 世代別
    df['期_数値'] = pd.to_numeric(df['入学期（◯期生）'], errors='coerce')
    def get_gen(ki):
        if pd.isna(ki): return None
        if ki <= 10: return '1〜10期'
        elif ki <= 20: return '11〜20期'
        elif ki <= 30: return '21〜30期'
        else: return '31〜37期'
    df['世代'] = df['期_数値'].apply(get_gen)

    print("\n  世代別全体満足度（補完後）:")
    for gen in ['1〜10期', '11〜20期', '21〜30期', '31〜37期']:
        g = df[df['世代'] == gen]
        if len(g) > 0:
            print(f"    {gen}: {g['全体_補完'].mean():.2f}")

    # 結果を保存
    result_df = df[['全体_数値', '全体_補完', 'gemini_sentiment', 'gemini_reasoning', 'engagement_score', '世代']].copy()
    result_df.to_csv('satisfaction_imputation_results.csv', index=False, encoding='utf-8')
    print("\n  結果をsatisfaction_imputation_results.csvに保存しました")

    # Geminiスコアの分布も表示
    if valid_gemini > 0:
        print(f"\n  Geminiスコア分布:")
        for score in range(1, 6):
            count = (df['gemini_sentiment'] == score).sum()
            print(f"    {score}点: {count}件")

if __name__ == '__main__':
    main()
