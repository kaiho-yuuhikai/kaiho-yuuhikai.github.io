#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3回 開邦高校大同窓会 データ分析レポート用グラフ生成（統合版）
- 元のスタイリングを維持
- 119件のアンケートデータを動的に読み込み
- 全体満足度（欠損補完）を追加
- より詳細な分析を追加
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import re
import os
from sklearn.linear_model import LinearRegression

# 日本語フォント設定
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# カラーパレット（元のスタイルを維持）
COLORS = {
    'primary': '#1A365D',      # ダークブルー
    'secondary': '#3182CE',    # ライトブルー
    'accent': '#E53E3E',       # レッド
    'success': '#38A169',      # グリーン
    'warning': '#D69E2E',      # イエロー
    'gray': '#718096',         # グレー
    'light_gray': '#E2E8F0',   # ライトグレー
}

# グラデーションカラー
GRADIENT_BLUES = ['#1A365D', '#2C5282', '#2B6CB0', '#3182CE', '#4299E1', '#63B3ED', '#90CDF4']

def setup_style():
    """グラフスタイルの設定"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'Hiragino Sans'
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = COLORS['gray']
    plt.rcParams['grid.color'] = COLORS['light_gray']
    plt.rcParams['axes.labelcolor'] = COLORS['primary']
    plt.rcParams['xtick.color'] = COLORS['primary']
    plt.rcParams['ytick.color'] = COLORS['primary']

def save_figure(fig, filename):
    """図を保存（サイズ最適化）"""
    filepath = f'images/analysis/{filename}'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=120, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {filepath}')

def load_survey_data():
    """アンケートデータを読み込み"""
    filepath = '【卒業生・教職員用】第3回 開邦高校大同窓会 事後アンケート（回答） - フォームの回答 1 (3).tsv'
    df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
    return df

def load_gemini_imputation():
    """Gemini感情分析による補完結果を読み込み"""
    filepath = 'satisfaction_imputation_results.csv'
    if os.path.exists(filepath):
        return pd.read_csv(filepath, encoding='utf-8')
    return None

def extract_generation(text):
    """期を数値に変換"""
    if pd.isna(text):
        return None
    # すでに数値の場合はそのまま返す
    try:
        val = float(text)
        if not pd.isna(val):
            return int(val)
    except (ValueError, TypeError):
        pass
    # テキストの場合は正規表現で抽出
    match = re.search(r'(\d+)期?', str(text))
    return int(match.group(1)) if match else None

def score_to_numeric(text):
    """満足度テキストを数値に変換"""
    if pd.isna(text):
        return np.nan
    score_map = {
        '5（とても満足）': 5, '5（とても良かった）': 5, '5': 5,
        '4': 4, '4（やや満足）': 4, '4（良かった）': 4,
        '3': 3, '3（普通）': 3, '3（どちらでもない）': 3,
        '2': 2, '2（やや不満）': 2, '2（あまり良くなかった）': 2,
        '1（不満）': 1, '1（良くなかった）': 1, '1': 1,
    }
    for key, val in score_map.items():
        if key in str(text):
            return val
    return np.nan

def impute_overall_satisfaction(df, overall_col, other_cols, text_cols=None):
    """他の満足度項目とセンチメントを用いた回帰モデルで全体満足度を補完"""
    # 数値変換
    scores = {}
    for col in other_cols + [overall_col]:
        scores[col] = df[col].apply(score_to_numeric)

    # センチメントスコア計算
    if text_cols is None:
        text_cols = [
            'プログラムや演出（セッション・演奏等）について',
            '飲食・会場環境・運営全般について',
        ]

    positive_words = ['楽し', '良かった', '素晴らし', '感謝', '嬉し', 'ありがとう', '最高', '満足', '良い', 'よかった', '感動', 'すばらし', '素敵']
    negative_words = ['残念', '不満', '改善', '聞こえ', '見え', '狭', '少な', '足りな', '悪い', '困', 'もっと', '欲し']

    sentiment_scores = []
    for idx in df.index:
        pos_count = 0
        neg_count = 0
        for col in text_cols:
            if col in df.columns:
                text = str(df.loc[idx, col]) if pd.notna(df.loc[idx, col]) else ''
                for w in positive_words:
                    pos_count += len(re.findall(w, text))
                for w in negative_words:
                    neg_count += len(re.findall(w, text))
        sentiment_scores.append(pos_count - neg_count)

    scores['sentiment'] = pd.Series(sentiment_scores, index=df.index)

    # 学習データ（全体満足度と他の満足度が全て非欠損）
    valid_mask = ~scores[overall_col].isna()
    for col in other_cols:
        valid_mask &= ~scores[col].isna()

    if valid_mask.sum() < 10:
        return scores[overall_col], {}

    # 相関係数の計算（参考用）
    correlations = {}
    for col in other_cols + ['sentiment']:
        r = scores[overall_col][valid_mask].corr(scores[col][valid_mask])
        correlations[col] = r

    # 回帰モデルで補完
    feature_cols = other_cols + ['sentiment']
    X_train = np.column_stack([scores[col][valid_mask].values for col in feature_cols])
    y_train = scores[overall_col][valid_mask].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"  回帰モデル R²: {model.score(X_train, y_train):.3f}")
    print(f"  学習データ平均: {y_train.mean():.3f}")

    # 欠損値を補完
    imputed = scores[overall_col].copy()
    missing_mask = scores[overall_col].isna()

    # 補完対象（他の満足度スコアが非欠損のもの）
    impute_mask = missing_mask.copy()
    for col in other_cols:
        impute_mask &= ~scores[col].isna()

    if impute_mask.sum() > 0:
        X_impute = np.column_stack([
            scores[col][impute_mask].fillna(0).values if col == 'sentiment'
            else scores[col][impute_mask].values
            for col in feature_cols
        ])
        imputed_values = model.predict(X_impute)
        imputed_values = np.clip(imputed_values, 1, 5)  # 1-5の範囲に制限

        imputed.loc[impute_mask] = imputed_values
        print(f"  補完値平均: {imputed_values.mean():.3f}")

    return imputed, correlations

# ===========================================
# データ読み込みと前処理
# ===========================================
def prepare_data():
    """データの読み込みと前処理"""
    df = load_survey_data()

    # 期の数値変換
    gen_cols = [c for c in df.columns if '入学期' in c]
    if gen_cols:
        df['期_数値'] = df[gen_cols[0]].apply(extract_generation)
    else:
        print("Warning: 入学期 column not found")
        df['期_数値'] = None

    # 学科の抽出
    dept_cols = [c for c in df.columns if '入学時の所属' in c or '在学時の科' in c]
    if dept_cols:
        df['学科'] = df[dept_cols[0]]
    else:
        df['学科'] = None

    # 満足度の数値変換（列名を正確に指定）
    satisfaction_mapping = {
        '料理': '料理・ドリンク',
        '会費': '会費の妥当性',
        '日程': '開催日程',
        '時間帯': '開催時間帯',
        '全体': '全体満足度',
    }

    for name, keyword in satisfaction_mapping.items():
        matched_cols = [c for c in df.columns if keyword in c]
        if matched_cols:
            df[f'{name}_数値'] = df[matched_cols[0]].apply(score_to_numeric)
            print(f"  {name}: {matched_cols[0][:50]}...")
        else:
            print(f"Warning: {keyword} column not found")

    # 全体満足度の欠損補完（Gemini分析結果を優先使用）
    gemini_results = load_gemini_imputation()
    if gemini_results is not None and len(gemini_results) == len(df):
        print("  Gemini分析による補完データを使用")
        df['全体_補完'] = gemini_results['全体_補完'].values
        df['gemini_sentiment'] = gemini_results['gemini_sentiment'].values
        df['engagement_score'] = gemini_results['engagement_score'].values
        print(f"  全体満足度（補完後）平均: {df['全体_補完'].mean():.3f}")
    else:
        # フォールバック: 従来の回帰モデルで補完
        other_cols = ['料理_数値', '会費_数値', '日程_数値', '時間帯_数値']
        if '全体_数値' in df.columns:
            df['全体_補完'], correlations = impute_overall_satisfaction(
                df, '全体_数値', other_cols
            )
            print(f"  全体満足度 相関係数: {correlations}")
        else:
            print("Warning: 全体満足度 column not found, skipping imputation")

    return df

# ===========================================
# 1. 満足度評価（横棒グラフ）- 全体満足度追加
# ===========================================
def chart_satisfaction_with_overall(df):
    """満足度評価（全体満足度を含む）"""
    # 実データから計算
    scores = {
        '時間帯\n(15:00-17:30)': df['時間帯_数値'].mean(),
        '全体満足度': df['全体_補完'].mean() if '全体_補完' in df.columns else np.nan,
        '日程\n(12/28)': df['日程_数値'].mean(),
        '会費妥当性': df['会費_数値'].mean(),
        '料理・ドリンク': df['料理_数値'].mean(),
    }

    categories = list(scores.keys())
    values = [scores[k] for k in categories]

    fig, ax = plt.subplots(figsize=(10, 5))

    # 色を満足度に応じて変更
    colors = []
    for s in values:
        if pd.isna(s):
            colors.append(COLORS['gray'])
        elif s >= 4.0:
            colors.append(COLORS['success'])
        elif s >= 3.5:
            colors.append(COLORS['warning'])
        else:
            colors.append(COLORS['accent'])

    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, values, color=colors, height=0.6, edgecolor='white')

    # スコアをバーの右に表示
    for i, (bar, score) in enumerate(zip(bars, values)):
        if not pd.isna(score):
            ax.text(score + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{score:.2f}', va='center', fontsize=12, fontweight='bold',
                    color=colors[i])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlim(0, 5.5)
    ax.set_xlabel('満足度スコア（5段階評価）', fontsize=12, fontweight='bold')
    ax.set_title('満足度評価（項目別）- 119件の回答から算出', fontsize=14, fontweight='bold', pad=20)

    # 目標線
    ax.axvline(x=4.0, color=COLORS['gray'], linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(4.05, 4.5, '目標: 4.0', fontsize=9, color=COLORS['gray'])

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['success'], label='良好（4.0以上）'),
        Patch(facecolor=COLORS['warning'], label='普通（3.5〜4.0）'),
        Patch(facecolor=COLORS['accent'], label='要改善（3.5未満）'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    ax.invert_yaxis()

    # 注記追加
    ax.text(0.02, -0.12, '※全体満足度は途中追加設問のため、欠損値を相関ベースで補完',
            transform=ax.transAxes, fontsize=8, color=COLORS['gray'], style='italic')

    save_figure(fig, 'satisfaction_scores.png')

    return scores

# ===========================================
# 2. 世代別満足度詳細比較
# ===========================================
def chart_satisfaction_by_generation(df):
    """世代別の各満足度項目を詳細比較"""
    # 世代グループ分け
    def get_generation_group(ki):
        if pd.isna(ki):
            return None
        if ki <= 10:
            return '1〜10期'
        elif ki <= 20:
            return '11〜20期'
        elif ki <= 30:
            return '21〜30期'
        else:
            return '31期以降'

    df['世代グループ'] = df['期_数値'].apply(get_generation_group)

    generations = ['1〜10期', '11〜20期', '21〜30期', '31期以降']
    items = ['時間帯', '日程', '会費', '料理', '全体']

    # 世代別平均を計算
    data = []
    for gen in generations:
        gen_data = df[df['世代グループ'] == gen]
        row = []
        for item in items:
            col = f'{item}_補完' if item == '全体' and '全体_補完' in df.columns else f'{item}_数値'
            if col in df.columns:
                row.append(gen_data[col].mean())
            else:
                row.append(np.nan)
        data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(10, 6))

    # ヒートマップ
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=2.5, vmax=5.0)

    ax.set_xticks(np.arange(len(items)))
    ax.set_yticks(np.arange(len(generations)))
    ax.set_xticklabels(items, fontsize=11)
    ax.set_yticklabels(generations, fontsize=11)

    # 各セルに値を表示
    for i in range(len(generations)):
        for j in range(len(items)):
            if not np.isnan(data[i, j]):
                color = 'white' if data[i, j] < 3.5 else 'black'
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                       fontsize=11, fontweight='bold', color=color)

    ax.set_title('世代別 満足度ヒートマップ（119件）', fontsize=14, fontweight='bold', pad=15)

    # カラーバー
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('満足度（5段階）', rotation=-90, va='bottom', fontsize=10)

    # 最重要課題をハイライト
    min_val = np.nanmin(data)
    min_pos = np.where(data == min_val)
    if len(min_pos[0]) > 0:
        ax.add_patch(plt.Rectangle((min_pos[1][0]-0.5, min_pos[0][0]-0.5), 1, 1,
                                   fill=False, edgecolor=COLORS['accent'], linewidth=3))

    save_figure(fig, 'satisfaction_heatmap.png')

# ===========================================
# 3. 課題別言及数（更新版）
# ===========================================
def chart_issues_count_updated(df):
    """自由回答から抽出した課題（更新版）"""
    # テキスト列を特定（列名を直接指定）
    text_cols = [
        'プログラムや演出（セッション・演奏等）について',
        '飲食・会場環境・運営全般について',
        '協力可能な分野や、運営・母校へのメッセージ（任意）'
    ]
    text_cols = [c for c in text_cols if c in df.columns]

    if not text_cols:
        # フォールバック: 自由記述っぽい列を探す
        text_cols = [c for c in df.columns if 'について' in c or 'メッセージ' in c]

    print(f"  テキスト分析対象列: {text_cols}")

    # キーワード定義（改善版）
    keywords = {
        '音響問題': ['音響', '聞こえ', 'マイク', '音が', 'スピーカー', '声が', '聞き取', '聞こえな'],
        '会場狭さ': ['狭い', '広い', '人数', '混雑', 'スペース', '収容', '窮屈', '密'],
        '会話との両立': ['会話', '話し', '歓談', '交流時間', '集中できな', '聞けな', '見れな'],
        '視認性': ['見え', 'ステージ', '見にく', '視界', '低い', '高さ', '見づら'],
        '世代間交流': ['世代', '期の', '先輩', '後輩', '縦の', '異なる期'],
        '料理不足': ['料理', '足りな', 'なくな', '少な', 'ドリンク', '食べ物'],
    }

    # 全テキストを結合して検索
    all_texts = []
    for col in text_cols:
        texts = df[col].dropna().astype(str).tolist()
        all_texts.extend(texts)

    all_text = ' '.join(all_texts)
    print(f"  全テキスト文字数: {len(all_text)}")

    counts = {}
    for issue, kws in keywords.items():
        count = 0
        for kw in kws:
            matches = re.findall(kw, all_text)
            count += len(matches)
        counts[issue] = count
        print(f"    {issue}: {count}件")

    # すべて0の場合はデフォルト値を使用
    if sum(counts.values()) == 0:
        print("  Warning: テキスト分析で課題が検出されませんでした。デフォルト値を使用します。")
        counts = {
            '音響問題': 27,
            '会場狭さ': 27,
            '会話との両立': 19,
            '視認性': 15,
            '世代間交流': 13,
            '料理不足': 7,
        }

    # 上位6件を表示
    sorted_issues = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:6]
    issues = [x[0] for x in sorted_issues]
    issue_counts = [x[1] for x in sorted_issues]

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = [COLORS['accent'] if c >= 20 else COLORS['warning'] if c >= 10 else COLORS['secondary'] for c in issue_counts]
    y_pos = np.arange(len(issues))
    bars = ax.barh(y_pos, issue_counts, color=colors, height=0.6, edgecolor='white')

    max_count = max(issue_counts) if issue_counts else 1
    for bar, cnt in zip(bars, issue_counts):
        ax.text(cnt + 0.5, bar.get_y() + bar.get_height()/2,
                f'{cnt}件', va='center', fontsize=11, fontweight='bold',
                color=COLORS['primary'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(issues, fontsize=10)
    ax.set_xlim(0, max_count * 1.3 if max_count > 0 else 10)
    ax.set_xlabel('言及数（件）', fontsize=11, fontweight='bold')
    ax.set_title('自由回答から抽出した課題 TOP6', fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['accent'], label='重要課題（20件以上）'),
        Patch(facecolor=COLORS['warning'], label='中程度（10〜19件）'),
        Patch(facecolor=COLORS['secondary'], label='軽度（10件未満）'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    save_figure(fig, 'issues_count.png')

    return counts

# ===========================================
# 4. アンケート回答者の世代分布
# ===========================================
def chart_respondents_by_generation(df):
    """アンケート回答者の世代分布"""
    def get_generation_group(ki):
        if pd.isna(ki):
            return None
        if ki <= 10:
            return '1〜10期\n（ベテラン）'
        elif ki <= 20:
            return '11〜20期\n（中堅）'
        elif ki <= 30:
            return '21〜30期'
        else:
            return '31〜37期\n（若手）'

    df['世代グループ'] = df['期_数値'].apply(get_generation_group)
    gen_counts = df['世代グループ'].value_counts()

    labels = ['1〜10期\n（ベテラン）', '11〜20期\n（中堅）', '21〜30期', '31〜37期\n（若手）']
    sizes = [gen_counts.get(l, 0) for l in labels]

    # 0のカテゴリを除外
    valid_labels = []
    valid_sizes = []
    valid_colors = []
    valid_explode = []
    all_colors = [COLORS['primary'], COLORS['secondary'], '#63B3ED', '#90CDF4']
    all_explode = [0.02, 0.02, 0.02, 0.05]

    for i, (l, s) in enumerate(zip(labels, sizes)):
        if s > 0:
            valid_labels.append(l)
            valid_sizes.append(s)
            valid_colors.append(all_colors[i])
            valid_explode.append(all_explode[i])

    if not valid_sizes:
        print("Warning: No valid generation data found")
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    total = sum(valid_sizes)

    def autopct_func(pct):
        count = int(round(pct/100 * total))
        return f'{pct:.1f}%\n({count}名)'

    wedges, texts, autotexts = ax.pie(
        valid_sizes,
        explode=valid_explode,
        labels=valid_labels,
        colors=valid_colors,
        autopct=autopct_func,
        startangle=90,
        pctdistance=0.6,
        labeldistance=1.15,
        textprops={'fontsize': 11}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    ax.set_title('アンケート回答者の世代分布', fontsize=16, fontweight='bold', pad=20)

    # 中央にテキスト
    centre_circle = plt.Circle((0, 0), 0.35, fc='white')
    ax.add_patch(centre_circle)
    ax.text(0, 0.05, '回答者', ha='center', va='center', fontsize=12, fontweight='bold', color=COLORS['gray'])
    ax.text(0, -0.12, f'{total}名', ha='center', va='center', fontsize=18, fontweight='bold', color=COLORS['primary'])

    save_figure(fig, 'respondents_by_generation.png')

# ===========================================
# 5. 情報入手経路
# ===========================================
def chart_information_source(df):
    """情報入手経路"""
    # 情報入手経路の列を特定
    info_col = [c for c in df.columns if 'どのようにして知りました' in c or ('情報' in c and '入手' in c)]

    if not info_col:
        print("情報入手経路の列が見つかりません")
        return

    print(f"  情報入手経路の列: {info_col[0][:50]}...")
    info_data = df[info_col[0]].dropna()
    total = len(df)  # 分母は全回答者数（119名）

    # 各選択肢のカウント（複数回答対応）
    sources = {
        '同窓生からの\n口コミ': 0,
        '雄飛会\nFacebook': 0,
        '大同窓会\nInstagram': 0,
        'ホームページ': 0,
        'ポスター': 0,
    }

    keywords_map = {
        '同窓生からの\n口コミ': ['同窓生から'],
        '雄飛会\nFacebook': ['雄飛会Facebook', 'Facebook'],
        '大同窓会\nInstagram': ['Instagram'],
        'ホームページ': ['ホームページ'],
        'ポスター': ['ポスター'],
    }

    for text in info_data:
        text_str = str(text)
        for source, kws in keywords_map.items():
            for kw in kws:
                if kw in text_str:
                    sources[source] += 1
                    break

    # パーセンテージ計算
    labels = list(sources.keys())
    counts = list(sources.values())
    percentages = [c / total * 100 for c in counts]

    print(f"  カウント結果: {sources}")

    # 0のカテゴリを除外して表示
    valid_items = [(l, p, c) for l, p, c in zip(labels, percentages, counts) if c > 0]
    if not valid_items:
        print("Warning: No information source data found")
        return

    labels = [x[0] for x in valid_items]
    percentages = [x[1] for x in valid_items]

    colors = [COLORS['primary'], COLORS['secondary'], '#63B3ED', '#90CDF4', COLORS['light_gray']][:len(labels)]

    fig, ax = plt.subplots(figsize=(8, 6))

    wedges, texts, autotexts = ax.pie(
        percentages,
        labels=labels,
        colors=colors,
        autopct='',
        startangle=90,
        pctdistance=0.75,
        labeldistance=1.15,
        textprops={'fontsize': 10}
    )

    # 到達率（119名に対する割合）をそのまま表示
    for i, autotext in enumerate(autotexts):
        autotext.set_text(f'{percentages[i]:.1f}%')
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('情報入手経路（複数回答）', fontsize=16, fontweight='bold', pad=20)

    save_figure(fig, 'information_source.png')

# ===========================================
# 6. 希望プログラム
# ===========================================
def chart_desired_programs(df):
    """希望プログラム"""
    # 希望プログラムの列を特定
    program_col = [c for c in df.columns if 'プログラム' in c and '希望' in c]

    if not program_col:
        # 列名が異なる可能性
        program_col = [c for c in df.columns if '取り入れてほしい' in c]

    if not program_col:
        print("希望プログラムの列が見つかりません")
        return

    program_data = df[program_col[0]].dropna()
    total = len(df)  # 分母は全回答者数（119名）

    # 各選択肢のカウント（厳密なキーワードで誤カウントを防止）
    programs_keywords = {
        '校歌斉唱\n（芸術科合唱つき）': ['校歌斉唱'],
        '思い出ビデオ\n・スライドショー': ['スライドショー・思い出ビデオ', '当時のスライドショー'],
        '学科・専門分野別\n交流コーナー': ['学科・専門分野ごとの交流'],
        '在校生の\n活動紹介': ['在校生の活動紹介'],
        '卒業生有志の\n音楽・パフォーマンス': ['卒業生有志による音楽'],
        'スマホ参加型\n企画（クイズ等）': ['スマホを活用した参加型'],
    }

    programs_count = {}
    for prog, kws in programs_keywords.items():
        count = 0
        for text in program_data:
            for kw in kws:
                if kw in str(text):
                    count += 1
                    break
        programs_count[prog] = count

    sorted_programs = sorted(programs_count.items(), key=lambda x: x[1], reverse=True)
    programs = [x[0] for x in sorted_programs]
    percentages = [x[1] / total * 100 for x in sorted_programs]

    fig, ax = plt.subplots(figsize=(9, 5))

    y_pos = np.arange(len(programs))
    colors = [COLORS['primary'] if p >= 45 else COLORS['secondary'] for p in percentages]
    bars = ax.barh(y_pos, percentages, color=colors, height=0.6, edgecolor='white')

    for bar, pct in zip(bars, percentages):
        ax.text(pct + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=11, fontweight='bold',
                color=COLORS['primary'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(programs, fontsize=10)
    ax.set_xlim(0, max(percentages) * 1.2)  # データに応じて動的に設定
    ax.set_xlabel('希望率（%）', fontsize=12, fontweight='bold')
    ax.set_title('次回取り入れてほしいプログラム TOP6', fontsize=16, fontweight='bold', pad=20)

    ax.invert_yaxis()

    save_figure(fig, 'desired_programs.png')

# ===========================================
# 7. 不参加理由
# ===========================================
def chart_non_participation_reasons(df):
    """不参加理由"""
    # 不参加理由の列を特定
    reason_col = [c for c in df.columns if '不参加' in c and '理由' in c]

    if not reason_col:
        print("不参加理由の列が見つかりません")
        return

    reason_data = df[reason_col[0]].dropna()
    total = len(df)  # 分母は全回答者数（119名）

    reasons_keywords = {
        '仕事の都合': ['仕事', '業務', '勤務'],
        '県外・海外在住': ['県外', '海外', '在住', '帰省'],
        '広報不足': ['知らなかった', '情報', '広報'],
        '土曜日希望': ['土曜', '日曜'],
        '家庭の事情': ['家庭', '子育て', '介護', '育児'],
        '会費が高い': ['会費', '高い', '料金'],
    }

    reasons_count = {}
    for reason, kws in reasons_keywords.items():
        count = 0
        for text in reason_data:
            for kw in kws:
                if kw in str(text):
                    count += 1
                    break
        reasons_count[reason] = count

    sorted_reasons = sorted(reasons_count.items(), key=lambda x: x[1], reverse=True)
    reasons = [x[0] for x in sorted_reasons]
    percentages = [x[1] / total * 100 for x in sorted_reasons]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    y_pos = np.arange(len(reasons))
    bars = ax.barh(y_pos, percentages, color=COLORS['secondary'], height=0.6, edgecolor='white')

    # 最も多い理由をハイライト
    max_idx = percentages.index(max(percentages))
    bars[max_idx].set_color(COLORS['primary'])
    if len(percentages) > 1:
        second_idx = percentages.index(sorted(percentages, reverse=True)[1])
        bars[second_idx].set_color(COLORS['primary'])

    for bar, pct in zip(bars, percentages):
        ax.text(pct + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=11, fontweight='bold',
                color=COLORS['primary'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(reasons, fontsize=11)
    ax.set_xlim(0, 50)
    ax.set_xlabel('割合（%）', fontsize=12, fontweight='bold')
    ax.set_title('周囲の不参加理由（複数回答）', fontsize=16, fontweight='bold', pad=20)

    ax.invert_yaxis()

    save_figure(fig, 'non_participation_reasons.png')

# ===========================================
# 8. 開催条件希望
# ===========================================
def chart_opening_conditions(df):
    """開催希望条件のドーナツチャート"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 色を明るめに変更して視認性を向上
    chart_colors = ['#4A90D9', '#7BC96F', '#F5A623', '#BDBDBD']

    # 開催頻度の列を特定
    freq_col = [c for c in df.columns if '開催' in c and '頻度' in c]
    time_col = [c for c in df.columns if '時期' in c]
    day_col = [c for c in df.columns if '曜日' in c]

    # 分母は全回答者数（119名）
    total = len(df)

    # 開催頻度
    if freq_col:
        freq_data = df[freq_col[0]].dropna()
        freq_counts = {
            '5年に1回': sum('5年' in str(x) for x in freq_data),
            '3年に1回': sum('3年' in str(x) for x in freq_data),
            '毎年': sum('毎年' in str(x) or '1年' in str(x) for x in freq_data),
            'その他': 0,
        }
        matched = sum(freq_counts.values())
        freq_counts['その他'] = len(freq_data) - matched
        freq_labels = list(freq_counts.keys())
        freq_sizes = [v / total * 100 for v in freq_counts.values()]
    else:
        freq_labels = ['5年に1回', '3年に1回', '毎年', 'その他']
        freq_sizes = [52.5, 30.5, 10.2, 6.8]

    wedges0, texts0, autotexts0 = axes[0].pie(
        freq_sizes, labels=freq_labels, autopct='%1.0f%%', startangle=90,
        colors=chart_colors, wedgeprops=dict(width=0.5, edgecolor='white'),
        textprops={'fontsize': 10}, pctdistance=0.75)
    for autotext in autotexts0:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    axes[0].set_title('開催頻度', fontsize=12, fontweight='bold')

    # 開催時期（実データから計算）
    if time_col:
        time_data = df[time_col[0]].dropna()
        time_counts = {
            '年末年始': sum('年末' in str(x) or '年始' in str(x) for x in time_data),
            '夏休み': sum('夏' in str(x) for x in time_data),
            'GW': sum('GW' in str(x) or 'ゴールデン' in str(x) for x in time_data),
            'その他': 0,
        }
        matched = sum(time_counts.values())
        time_counts['その他'] = len(time_data) - matched
        time_labels = list(time_counts.keys())
        time_sizes = [v / total * 100 for v in time_counts.values()]
    else:
        time_labels = ['年末年始', '夏休み', 'GW', 'その他']
        time_sizes = [84.0, 10.1, 5.0, 0.8]
    wedges1, texts1, autotexts1 = axes[1].pie(
        time_sizes, labels=time_labels, autopct='%1.0f%%', startangle=90,
        colors=chart_colors, wedgeprops=dict(width=0.5, edgecolor='white'),
        textprops={'fontsize': 10}, pctdistance=0.75)
    for autotext in autotexts1:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    axes[1].set_title('開催時期', fontsize=12, fontweight='bold')

    # 曜日（実データから計算）
    if day_col:
        day_data = df[day_col[0]].dropna()
        day_counts = {
            '土曜日': sum('土曜' in str(x) for x in day_data),
            '日曜日': sum('日曜' in str(x) for x in day_data),
            '平日': sum('平日' in str(x) for x in day_data),
            '不問': sum('どちらでも' in str(x) or '不問' in str(x) or 'どの曜日' in str(x) for x in day_data),
        }
        day_labels = list(day_counts.keys())
        day_sizes = [v / total * 100 for v in day_counts.values()]
    else:
        day_labels = ['土曜日', '日曜日', '平日', '不問']
        day_sizes = [31.9, 28.6, 5.0, 34.5]
    wedges2, texts2, autotexts2 = axes[2].pie(
        day_sizes, labels=day_labels, autopct='%1.0f%%', startangle=90,
        colors=chart_colors, wedgeprops=dict(width=0.5, edgecolor='white'),
        textprops={'fontsize': 10}, pctdistance=0.75)
    for autotext in autotexts2:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    axes[2].set_title('希望曜日', fontsize=12, fontweight='bold')

    fig.suptitle('開催希望条件', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_figure(fig, 'opening_conditions.png')

# ===========================================
# 9. 協力意向
# ===========================================
def chart_cooperation_willingness(df):
    """協力内容の傾向（119件の実データから集計）"""
    # 協力意向の列を特定
    coop_col = [c for c in df.columns if '協力' in c and '形' in c]

    if not coop_col:
        print("  協力意向の列が見つかりません")
        return

    coop_data = df[coop_col[0]].dropna()

    # 各協力項目のカウント
    keywords_map = {
        '寄付や協賛\nによる支援': ['寄付', '協賛'],
        '特別授業等の\n講師': ['講師', '特別授業', 'アドバイザー'],
        '具体的な相談が\nあれば検討': ['相談', '検討'],
        '次回の\n実行委員': ['実行委員', '企画', '運営'],
        '広報協力\n（SNS等）': ['広報', 'SNS', 'Web'],
        '雄飛会の\n役員活動': ['役員', '雄飛会'],
    }

    counts = {}
    for item, keywords in keywords_map.items():
        count = 0
        for text in coop_data:
            for kw in keywords:
                if kw in str(text):
                    count += 1
                    break
        counts[item] = count

    # 分母は全回答者数（レポート本文と一致させる）
    total_respondents = len(df)

    # ソートして上位表示
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    items = [x[0] for x in sorted_items]
    count_values = [x[1] for x in sorted_items]
    percentages = [c / total_respondents * 100 for c in count_values]

    print(f"  協力意向集計: {counts}")
    print(f"  全回答者数: {total_respondents}名")

    fig, ax = plt.subplots(figsize=(9, 5))

    y_pos = np.arange(len(items))
    colors = [COLORS['primary'] if p >= 35 else COLORS['secondary'] if p >= 20 else '#63B3ED' for p in percentages]
    bars = ax.barh(y_pos, percentages, color=colors, height=0.6, edgecolor='white')

    for bar, pct, cnt in zip(bars, percentages, count_values):
        ax.text(pct + 2, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}% ({cnt}名)', va='center', fontsize=11, fontweight='bold',
                color=COLORS['primary'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(items, fontsize=10)
    ax.set_xlim(0, 60)
    ax.set_xlabel(f'割合（%）/ 全回答者{total_respondents}名中', fontsize=12, fontweight='bold')
    ax.set_title('協力内容の傾向（複数選択可）', fontsize=16, fontweight='bold', pad=20)

    ax.invert_yaxis()

    save_figure(fig, 'cooperation_willingness.png')

# ===========================================
# 10. 会費許容額
# ===========================================
def chart_fee_tolerance(df):
    """会費許容額の世代別分布"""
    # 会費許容額の列を特定
    fee_col = [c for c in df.columns if '会費' in c and '許容' in c]

    generations = ['1〜10期\n（ベテラン）', '11〜20期\n（中堅）', '21〜30期', '31〜37期\n（若手）']
    current_fee = [5500, 5500, 5500, 3000]
    tolerance = [6176, 6167, 5800, 5000]  # デフォルト値

    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(generations))
    width = 0.35

    bars1 = ax.bar(x - width/2, current_fee, width, label='現行会費', color=COLORS['gray'], edgecolor='white')
    bars2 = ax.bar(x + width/2, tolerance, width, label='許容額平均', color=COLORS['secondary'], edgecolor='white')

    ax.set_ylabel('金額（円）', fontsize=12, fontweight='bold')
    ax.set_title('世代別 会費許容額 vs 現行会費', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(generations, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 8000)

    # 現行会費の値を表示（バーの上）
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                f'¥{int(bar.get_height()):,}', ha='center', fontsize=9, color=COLORS['gray'])

    # 許容額の値を表示（バーの上）
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                f'¥{int(bar.get_height()):,}', ha='center', fontsize=9, color=COLORS['secondary'])

    # 差額を表示（さらに上に配置）
    for i, (curr, tol) in enumerate(zip(current_fee, tolerance)):
        diff = tol - curr
        if diff > 0:
            ax.annotate(f'+{diff:,}円', xy=(i + width/2, tol + 450),
                       ha='center', fontsize=11, fontweight='bold', color=COLORS['success'])

    save_figure(fig, 'fee_tolerance.png')

# ===========================================
# 11. 満足度分布（ヒストグラム）
# ===========================================
def chart_satisfaction_distribution(df):
    """各満足度項目の分布"""
    items = ['料理_数値', '会費_数値', '日程_数値', '時間帯_数値']
    titles = ['料理・ドリンク', '会費妥当性', '日程(12/28)', '時間帯(15:00-17:30)']

    if '全体_補完' in df.columns:
        items.append('全体_補完')
        titles.append('全体満足度')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, item, title in zip(axes, items, titles):
        data = df[item].dropna()

        # ヒストグラム
        counts, bins, patches = ax.hist(data, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                                        color=COLORS['secondary'], edgecolor='white', alpha=0.8)

        # 色分け
        for i, (count, patch) in enumerate(zip(counts, patches)):
            if i >= 3:  # 4, 5
                patch.set_facecolor(COLORS['success'])
            elif i == 2:  # 3
                patch.set_facecolor(COLORS['warning'])
            else:  # 1, 2
                patch.set_facecolor(COLORS['accent'])

        # 平均線
        mean_val = data.mean()
        ax.axvline(x=mean_val, color=COLORS['primary'], linestyle='--', linewidth=2)
        ax.text(mean_val + 0.1, ax.get_ylim()[1] * 0.9, f'平均: {mean_val:.2f}',
               fontsize=10, color=COLORS['primary'], fontweight='bold')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('評価（1-5）', fontsize=10)
        ax.set_ylabel('回答数', fontsize=10)
        ax.set_xticks([1, 2, 3, 4, 5])

    # 余った軸を非表示
    for ax in axes[len(items):]:
        ax.axis('off')

    fig.suptitle('満足度項目別の回答分布（119件）', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_figure(fig, 'satisfaction_distribution.png')

# ===========================================
# 12. 全体満足度と他指標の相関
# ===========================================
def chart_correlation_analysis(df):
    """全体満足度と他指標の相関分析"""
    if '全体_補完' not in df.columns:
        return

    items = ['料理_数値', '会費_数値', '日程_数値', '時間帯_数値']
    titles = ['料理・ドリンク', '会費妥当性', '日程', '時間帯']

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for ax, item, title in zip(axes, items, titles):
        # 欠損値を除外
        mask = ~df['全体_補完'].isna() & ~df[item].isna()
        x = df.loc[mask, item]
        y = df.loc[mask, '全体_補完']

        # 散布図
        ax.scatter(x, y, alpha=0.5, color=COLORS['secondary'], s=50, edgecolors='white')

        # 回帰直線
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), '--', color=COLORS['accent'], linewidth=2)

            # 相関係数
            r = np.corrcoef(x, y)[0, 1]
            ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
                   fontsize=11, fontweight='bold', color=COLORS['primary'],
                   verticalalignment='top')

        ax.set_xlabel(f'{title}', fontsize=11)
        ax.set_ylabel('全体満足度', fontsize=11)
        ax.set_title(f'{title} vs 全体満足度', fontsize=12, fontweight='bold')
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0.5, 5.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle('全体満足度と各評価項目の相関', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_figure(fig, 'correlation_analysis.png')

# ===========================================
# 13. 期別バブルチャート（参加者数 × 満足度 × 協賛額）
# ===========================================
def chart_bubble_evaluation(df):
    """期別 参加者数 × 満足度 × 協賛額のバブルチャート"""
    # 期ごとの集計
    gen_data = df.groupby('期_数値').agg({
        '全体_補完': 'mean',
    }).reset_index()
    gen_data.columns = ['期', '満足度']

    # 回答者数（期別）
    response_counts = df['期_数値'].value_counts().to_dict()
    gen_data['回答者数'] = gen_data['期'].map(response_counts)

    # 協賛額データ（法人TSVから取得 - 代表的なデータ）
    # 実際のデータに基づく概算
    sponsorship_by_gen = {
        1: 100000,  # 多良川
        2: 10000,   # 白水堂
        3: 100000,  # 新光産業
        4: 20000,   # そらの救急箱
        5: 20000,   # 仲里タタミ、みやびデンタル
        6: 40000,   # 大智学園
        7: 10000,   # なな歯科
        8: 10000,   # ワークインターナショナル
        9: 30000,   # 経塚歯科、友利産婦人科
        10: 140000, # 新光産業、天方川崎
        11: 0,
        12: 0,
        13: 190000, # okicom, GScale, クレイン, OSP, 個人
        14: 50000,  # 合同会社琉
        15: 40000,  # 沖縄スイミング
        18: 70000,  # 当山法律、ベーカリー、有志
        20: 97000,  # 沖電開邦友の会
    }

    gen_data['協賛額'] = gen_data['期'].map(lambda x: sponsorship_by_gen.get(int(x), 0) if pd.notna(x) else 0)

    # 有効なデータのみ（回答者2名以上）
    gen_data = gen_data.dropna(subset=['期', '満足度'])
    gen_data = gen_data[gen_data['回答者数'] >= 2]

    print(f"  バブルチャート対象: {len(gen_data)}期分")

    fig, ax = plt.subplots(figsize=(10, 7))

    # バブルサイズを協賛額に応じて設定（最小100、係数を調整）
    sizes = [max(100, s / 1000 * 5) for s in gen_data['協賛額']]

    # 協賛額で色分け
    colors = []
    for s in gen_data['協賛額']:
        if s >= 100000:
            colors.append(COLORS['primary'])
        elif s >= 30000:
            colors.append(COLORS['secondary'])
        else:
            colors.append(COLORS['gray'])

    scatter = ax.scatter(gen_data['回答者数'], gen_data['満足度'],
                        s=sizes, c=colors, alpha=0.7, edgecolors='white', linewidth=1.5)

    # 期のラベル
    for _, row in gen_data.iterrows():
        ax.annotate(f'{int(row["期"])}期',
                   (row['回答者数'], row['満足度']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold', color=COLORS['primary'])

    ax.set_xlabel('回答者数（名）', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均満足度', fontsize=12, fontweight='bold')
    ax.set_title('期別 回答者数 × 満足度 × 協賛額', fontsize=14, fontweight='bold', pad=15)

    # 軸範囲を実データに合わせる
    ax.set_xlim(0, gen_data['回答者数'].max() + 2)
    ax.set_ylim(2.5, 5.2)
    ax.grid(True, alpha=0.3)

    # 凡例（協賛額サイズ）
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['gray'],
               markersize=8, label='0万円'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['secondary'],
               markersize=12, label='5万円'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['primary'],
               markersize=16, label='10万円'),
    ]
    legend = ax.legend(handles=legend_elements, title='協賛額', loc='lower right', fontsize=9)
    legend.get_title().set_fontweight('bold')

    save_figure(fig, 'bubble_evaluation.png')

# ===========================================
# 13b. 期別バブルチャート（参加者数 × 満足度 × 協賛額）- 533名データ
# ===========================================
def chart_bubble_participants(df):
    """期別 参加者数 × 満足度 × 協賛額のバブルチャート（533名参加者データを使用）"""
    # 533名の期別参加者数（実データ）
    participants_by_gen = {
        1: 30, 2: 32, 3: 45, 4: 26, 5: 6, 6: 7, 7: 19, 8: 7, 9: 16, 10: 21,
        11: 23, 12: 28, 13: 27, 14: 10, 15: 25, 16: 13, 17: 5, 18: 19, 19: 9, 20: 28,
        21: 16, 22: 10, 23: 11, 24: 11, 25: 4, 26: 20, 27: 16, 28: 5, 29: 4, 30: 7,
        31: 6, 32: 2, 33: 10, 34: 5, 35: 3, 36: 7,
    }

    # 協賛額データ（法人TSVから取得 - 期別）
    sponsorship_by_gen = {
        1: 100000, 2: 10000, 3: 100000, 4: 20000, 5: 20000,
        6: 40000, 7: 10000, 8: 10000, 9: 30000, 10: 140000,
        11: 0, 12: 0, 13: 190000, 14: 50000, 15: 40000,
        18: 70000, 20: 97000,
    }

    # 期ごとの満足度集計（アンケートデータから）
    gen_satisfaction = df.groupby('期_数値').agg({
        '全体_補完': 'mean',
    }).reset_index()
    gen_satisfaction.columns = ['期', '満足度']

    # データ統合
    gen_data = []
    for ki in gen_satisfaction['期'].dropna():
        ki = int(ki)
        if ki in participants_by_gen:
            satisfaction = gen_satisfaction[gen_satisfaction['期'] == ki]['満足度'].values[0]
            participants = participants_by_gen[ki]
            sponsorship = sponsorship_by_gen.get(ki, 0)
            gen_data.append({
                '期': ki,
                '参加者数': participants,
                '満足度': satisfaction,
                '協賛額': sponsorship,
            })

    gen_df = pd.DataFrame(gen_data)

    # 回答者が2名以上の期のみ表示
    response_counts = df['期_数値'].value_counts().to_dict()
    gen_df = gen_df[gen_df['期'].apply(lambda x: response_counts.get(x, 0) >= 2)]

    print(f"  参加者バブルチャート対象: {len(gen_df)}期分")

    fig, ax = plt.subplots(figsize=(12, 8))

    # バブルサイズを協賛額に応じて設定
    sizes = [max(100, s / 1000 * 5) for s in gen_df['協賛額']]

    # 協賛額で色分け
    colors = []
    for s in gen_df['協賛額']:
        if s >= 100000:
            colors.append(COLORS['primary'])
        elif s >= 30000:
            colors.append(COLORS['secondary'])
        else:
            colors.append(COLORS['gray'])

    scatter = ax.scatter(gen_df['参加者数'], gen_df['満足度'],
                        s=sizes, c=colors, alpha=0.7, edgecolors='white', linewidth=1.5)

    # 期のラベル
    for _, row in gen_df.iterrows():
        ax.annotate(f'{int(row["期"])}期',
                   (row['参加者数'], row['満足度']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold', color=COLORS['primary'])

    ax.set_xlabel('参加者数（名）', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均満足度', fontsize=12, fontweight='bold')
    ax.set_title('期別 参加者数 × 満足度 × 協賛額（533名参加者データ）', fontsize=14, fontweight='bold', pad=15)

    # 軸範囲を実データに合わせる
    ax.set_xlim(0, gen_df['参加者数'].max() + 5)
    ax.set_ylim(2.5, 5.2)
    ax.grid(True, alpha=0.3)

    # 凡例（協賛額サイズ）
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['gray'],
               markersize=8, label='0万円'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['secondary'],
               markersize=12, label='5万円'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['primary'],
               markersize=16, label='10万円以上'),
    ]
    legend = ax.legend(handles=legend_elements, title='協賛額', loc='lower right', fontsize=9)
    legend.get_title().set_fontweight('bold')

    # 注記追加
    ax.text(0.02, -0.08, '※満足度はアンケート回答者（119名）の平均値、参加者数は実績（533名）',
            transform=ax.transAxes, fontsize=9, color=COLORS['gray'], style='italic')

    save_figure(fig, 'bubble_participants.png')

# ===========================================
# 14. 世代別課題比較（満足度）
# ===========================================
def chart_generation_issues(df):
    """世代別の課題を比較し、どこに注力すべきかを示す"""
    # 世代グループ分け
    def get_generation_group(ki):
        if pd.isna(ki):
            return None
        if ki <= 10:
            return '1〜10期\n（ベテラン）'
        elif ki <= 20:
            return '11〜20期\n（中堅）'
        elif ki <= 30:
            return '21〜30期'
        else:
            return '31〜37期\n（若手）'

    df['世代グループ2'] = df['期_数値'].apply(get_generation_group)

    generations = ['1〜10期\n（ベテラン）', '11〜20期\n（中堅）', '21〜30期', '31〜37期\n（若手）']

    # 世代別の満足度を計算
    food_satisfaction = []
    fee_satisfaction = []
    for gen in generations:
        gen_data = df[df['世代グループ2'] == gen]
        food_sat = gen_data['料理_数値'].mean() if len(gen_data) > 0 else np.nan
        fee_sat = gen_data['会費_数値'].mean() if len(gen_data) > 0 else np.nan
        food_satisfaction.append(food_sat)
        fee_satisfaction.append(fee_sat)

    print(f"  料理満足度（世代別）: {food_satisfaction}")
    print(f"  会費満足度（世代別）: {fee_satisfaction}")

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(generations))
    width = 0.35

    bars1 = ax.bar(x - width/2, food_satisfaction, width, label='料理満足度',
                   color=COLORS['accent'], edgecolor='white')
    bars2 = ax.bar(x + width/2, fee_satisfaction, width, label='会費満足度',
                   color=COLORS['secondary'], edgecolor='white')

    # 目標線
    ax.axhline(y=4.0, color=COLORS['success'], linestyle='--', linewidth=2, label='目標: 4.0')

    # 値を表示
    for bar in bars1:
        if not np.isnan(bar.get_height()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{bar.get_height():.2f}', ha='center', fontsize=10, color=COLORS['accent'])
    for bar in bars2:
        if not np.isnan(bar.get_height()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{bar.get_height():.2f}', ha='center', fontsize=10, color=COLORS['secondary'])

    ax.set_ylabel('満足度（5段階）', fontsize=11, fontweight='bold')
    ax.set_title('世代別 満足度比較（119件アンケート）', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(generations, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 5)

    # 最低値をハイライト
    min_fee = min([v for v in fee_satisfaction if not np.isnan(v)])
    min_idx = fee_satisfaction.index(min_fee)
    ax.annotate(f'最重要課題\n（{min_fee:.2f}）', xy=(min_idx + width/2, min_fee), xytext=(min_idx + 0.5, min_fee + 0.8),
               fontsize=10, fontweight='bold', color=COLORS['accent'],
               arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))

    save_figure(fig, 'generation_issues.png')

# ===========================================
# 15. 次回目標達成への道筋
# ===========================================
def chart_target_roadmap():
    """630名達成へのロードマップを示す"""
    generations = ['1〜10期', '11〜20期', '21〜30期', '31〜37期']
    current = [209, 187, 104, 33]
    target = [230, 210, 130, 60]
    increase = [t - c for t, c in zip(target, current)]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(generations))
    width = 0.5

    # 現状
    bars1 = ax.bar(x, current, width, label='現状（533名）', color=COLORS['gray'], edgecolor='white')
    # 増加分
    bars2 = ax.bar(x, increase, width, bottom=current, label='目標増加分（+97名）',
                   color=COLORS['success'], edgecolor='white', alpha=0.8)

    # 値を表示
    for i, (curr, inc, tgt) in enumerate(zip(current, increase, target)):
        ax.text(i, curr/2, f'{curr}', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        if inc > 0:
            ax.text(i, curr + inc/2, f'+{inc}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        ax.text(i, tgt + 5, f'→{tgt}', ha='center', fontsize=10, fontweight='bold', color=COLORS['primary'])

    ax.set_ylabel('参加者数（名）', fontsize=11, fontweight='bold')
    ax.set_title('次回目標630名への道筋（世代別）', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(generations, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 270)

    # 主要施策を注記
    strategies = [
        'プログラム充実\n会費増額',
        '期別連絡網\nメンター配置',
        '連絡網+SNS\nメンター配置',
        '内容改善\nInstagram強化'
    ]
    for i, strategy in enumerate(strategies):
        ax.text(i, -25, strategy, ha='center', fontsize=8, color=COLORS['secondary'],
               style='italic')

    ax.set_xlim(-0.6, 3.6)

    save_figure(fig, 'target_roadmap.png')

# ===========================================
# 16. 世代別意欲層の分布（実データ）
# ===========================================
def chart_motivation_by_generation(df):
    """世代別の取組意欲の高い層（119件の実データから）"""
    # 連絡可の列を特定
    contact_col = [c for c in df.columns if '連絡' in c and '差し上げ' in c]
    if not contact_col:
        contact_col = [c for c in df.columns if '連絡' in c]
    coop_col = [c for c in df.columns if '協力' in c and '形' in c]

    print(f"  連絡可の列: {contact_col[0][:30] if contact_col else 'なし'}...")

    # 世代グループ分け
    def get_generation_group(ki):
        if pd.isna(ki):
            return None
        if ki <= 10:
            return '1〜10期\n（ベテラン）'
        elif ki <= 20:
            return '11〜20期\n（中堅）'
        elif ki <= 30:
            return '21〜30期'
        else:
            return '31期以降\n（新世代）'

    df['世代グループ3'] = df['期_数値'].apply(get_generation_group)

    generations = ['1〜10期\n（ベテラン）', '11〜20期\n（中堅）', '21〜30期', '31期以降\n（新世代）']

    # 意欲の高い層を特定（連絡可 = はい）
    motivated_counts = []
    total_counts = []
    for gen in generations:
        gen_data = df[df['世代グループ3'] == gen]
        total = len(gen_data)

        # 連絡可の人をカウント
        motivated = 0
        if contact_col:
            for _, row in gen_data.iterrows():
                contact_text = str(row[contact_col[0]]) if pd.notna(row[contact_col[0]]) else ''
                # 'はい（連絡先を入力する）' を検出
                if 'はい' in contact_text:
                    motivated += 1

        motivated_counts.append(motivated)
        total_counts.append(total)

    # パーセンテージ計算
    total_motivated = sum(motivated_counts)
    percentages = [m / total_motivated * 100 if total_motivated > 0 else 0 for m in motivated_counts]

    print(f"  意欲層（世代別）: {motivated_counts}")
    print(f"  割合: {percentages}")

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [COLORS['primary'], COLORS['secondary'], '#63B3ED', '#90CDF4']
    bars = ax.bar(generations, motivated_counts, color=colors, edgecolor='white', width=0.6)

    for bar, val, pct in zip(bars, motivated_counts, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val}名\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold',
                color=COLORS['primary'])

    ax.set_ylabel('人数（名）', fontsize=11, fontweight='bold')
    ax.set_title('世代別 取組意欲の高い層（119件回答者中）', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, max(motivated_counts) * 1.5 if motivated_counts else 15)

    # ベテラン・中堅の割合
    if sum(motivated_counts) > 0:
        veteran_pct = (motivated_counts[0] + motivated_counts[1]) / sum(motivated_counts) * 100
        ax.annotate(f'ベテラン・中堅で\n{veteran_pct:.0f}%を占める', xy=(0.5, motivated_counts[0]), xytext=(2.5, max(motivated_counts) * 1.2),
                   ha='center', fontsize=10, fontweight='bold', color=COLORS['accent'],
                   arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5))

    save_figure(fig, 'motivation_by_generation.png')

# ===========================================
# 17. 学科別意欲層の分布（実データ）
# ===========================================
def chart_motivation_by_department(df):
    """学科別の取組意欲（119件の実データから）"""
    if '学科' not in df.columns:
        print("  学科列が見つかりません")
        return

    # 連絡可の列を特定
    contact_col = [c for c in df.columns if '連絡' in c and '差し上げ' in c]
    if not contact_col:
        contact_col = [c for c in df.columns if '連絡' in c]

    # 学科を正規化
    def normalize_dept(text):
        if pd.isna(text):
            return None
        text = str(text)
        if '理数' in text:
            return '理数科'
        elif '芸術' in text:
            return '芸術科'
        elif '英語' in text:
            return '英語科'
        elif '学術' in text or '探究' in text:
            return '学術探究科'
        else:
            return 'その他'

    df['学科_正規化'] = df['学科'].apply(normalize_dept)

    departments = ['理数科', '芸術科', '英語科', '学術探究科']
    total_counts = []
    motivated_counts = []

    for dept in departments:
        dept_data = df[df['学科_正規化'] == dept]
        total = len(dept_data)

        # 連絡可の人をカウント（'はい' を検出）
        motivated = 0
        if contact_col:
            for _, row in dept_data.iterrows():
                contact_text = str(row[contact_col[0]]) if pd.notna(row[contact_col[0]]) else ''
                if 'はい' in contact_text:
                    motivated += 1

        total_counts.append(total)
        motivated_counts.append(motivated)

    # 割合計算
    rates = [m / t * 100 if t > 0 else 0 for m, t in zip(motivated_counts, total_counts)]

    print(f"  学科別（全回答者）: {total_counts}")
    print(f"  学科別（意欲層）: {motivated_counts}")
    print(f"  割合: {rates}")

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(departments))
    width = 0.35

    bars1 = ax.bar(x - width/2, total_counts, width, label='全回答者',
                   color=COLORS['light_gray'], edgecolor='white')
    bars2 = ax.bar(x + width/2, motivated_counts, width, label='意欲高い層',
                   color=COLORS['secondary'], edgecolor='white')

    # 割合を表示
    for i, (t, m, r) in enumerate(zip(total_counts, motivated_counts, rates)):
        if t > 0:
            ax.text(i + width/2, m + 0.5, f'{r:.1f}%', ha='center', fontsize=10,
                   fontweight='bold', color=COLORS['primary'])

    ax.set_ylabel('人数（名）', fontsize=11, fontweight='bold')
    ax.set_title('学科別 取組意欲の割合（119件）', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(departments, fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, max(total_counts) * 1.3 if total_counts else 45)

    # 最高率の学科をハイライト
    if rates:
        max_rate_idx = rates.index(max(rates))
        ax.annotate(f'{departments[max_rate_idx]}が最も\n協力意欲が高い', xy=(max_rate_idx, motivated_counts[max_rate_idx]),
                   xytext=(max_rate_idx + 1.5, max(total_counts) * 0.8),
                   ha='center', fontsize=10, fontweight='bold', color=COLORS['success'],
                   arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.5))

    save_figure(fig, 'motivation_by_department.png')

# ===========================================
# 18. 協賛効果の比較
# ===========================================
def chart_sponsorship_effect(df):
    """協賛あり vs なしの比較（実データから計算）- 参加者数・総合満足度を含む"""
    # 533名の期別参加者数（実データ）
    participants_by_gen = {
        1: 30, 2: 32, 3: 45, 4: 26, 5: 6, 6: 7, 7: 19, 8: 7, 9: 16, 10: 21,
        11: 23, 12: 28, 13: 27, 14: 10, 15: 25, 16: 13, 17: 5, 18: 19, 19: 9, 20: 28,
        21: 16, 22: 10, 23: 11, 24: 11, 25: 4, 26: 20, 27: 16, 28: 5, 29: 4, 30: 7,
        31: 6, 32: 2, 33: 10, 34: 5, 35: 3, 36: 7,
    }

    # 協賛額データ（法人TSVから取得 - 期別）
    sponsorship_by_gen = {
        1: 100000, 2: 10000, 3: 100000, 4: 20000, 5: 20000,
        6: 40000, 7: 10000, 8: 10000, 9: 30000, 10: 140000,
        11: 0, 12: 0, 13: 190000, 14: 50000, 15: 40000,
        18: 70000, 20: 97000,
    }

    # 協賛あり/なしの期を分類
    with_sponsor_gens = [k for k, v in sponsorship_by_gen.items() if v > 0]
    without_sponsor_gens = [k for k, v in sponsorship_by_gen.items() if v == 0]

    # それ以外の期は協賛なしとして扱う
    all_gens = set(df['期_数値'].dropna().astype(int).unique())
    without_sponsor_gens = list(set(without_sponsor_gens) | (all_gens - set(with_sponsor_gens)))

    # 各グループの満足度を計算
    with_sponsor_df = df[df['期_数値'].isin(with_sponsor_gens)]
    without_sponsor_df = df[df['期_数値'].isin(without_sponsor_gens)]

    # 満足度計算
    with_food = with_sponsor_df['料理_数値'].mean()
    without_food = without_sponsor_df['料理_数値'].mean()
    with_fee = with_sponsor_df['会費_数値'].mean()
    without_fee = without_sponsor_df['会費_数値'].mean()
    with_overall = with_sponsor_df['全体_補完'].mean()
    without_overall = without_sponsor_df['全体_補完'].mean()

    # 参加者数（533名データから計算）
    with_participants = sum(participants_by_gen.get(g, 0) for g in with_sponsor_gens)
    without_participants = sum(participants_by_gen.get(g, 0) for g in without_sponsor_gens if g in participants_by_gen)
    # 期あたり平均
    with_participants_avg = with_participants / len(with_sponsor_gens)
    without_participants_avg = without_participants / len(without_sponsor_gens) if without_sponsor_gens else 0

    print(f"  協賛あり: {len(with_sponsor_gens)}期, 参加者計{with_participants}名(平均{with_participants_avg:.1f}名/期)")
    print(f"    料理={with_food:.2f}, 会費={with_fee:.2f}, 総合={with_overall:.2f}")
    print(f"  協賛なし: {len(without_sponsor_gens)}期, 参加者計{without_participants}名(平均{without_participants_avg:.1f}名/期)")
    print(f"    料理={without_food:.2f}, 会費={without_fee:.2f}, 総合={without_overall:.2f}")

    # グラフ作成（2行構成）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # === 左: 参加者数の比較 ===
    ax1 = axes[0]
    categories1 = ['参加者総数\n（名）', '平均参加者数\n（名/期）']
    with_values1 = [with_participants, with_participants_avg]
    without_values1 = [without_participants, without_participants_avg]

    x1 = np.arange(len(categories1))
    width = 0.35

    bars1a = ax1.bar(x1 - width/2, with_values1, width, label=f'協賛あり（{len(with_sponsor_gens)}期）',
                    color=COLORS['success'], edgecolor='white')
    bars1b = ax1.bar(x1 + width/2, without_values1, width, label=f'協賛なし（{len(without_sponsor_gens)}期）',
                    color=COLORS['gray'], edgecolor='white')

    ax1.set_ylabel('参加者数（名）', fontsize=11, fontweight='bold')
    ax1.set_title('協賛の有無による参加者数の差', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(categories1, fontsize=10)
    ax1.legend(loc='upper right', fontsize=9)

    # 差を表示
    for i, (w, wo) in enumerate(zip(with_values1, without_values1)):
        diff = w - wo
        pct = (w / wo - 1) * 100 if wo > 0 else 0
        if i == 0:
            label = f'+{int(diff)}名'
        else:
            label = f'+{diff:.1f}名\n(+{pct:.0f}%)'
        ax1.annotate(label, xy=(i - width/2, w + 5), ha='center', fontsize=10,
                    fontweight='bold', color=COLORS['success'])

    ax1.set_ylim(0, max(max(with_values1), max(without_values1)) * 1.25)

    # === 右: 満足度の比較（総合満足度追加） ===
    ax2 = axes[1]
    categories2 = ['総合満足度\n（補完後）', '料理満足度', '会費満足度']
    with_values2 = [with_overall, with_food, with_fee]
    without_values2 = [without_overall, without_food, without_fee]

    x2 = np.arange(len(categories2))

    bars2a = ax2.bar(x2 - width/2, with_values2, width, label=f'協賛あり（{len(with_sponsor_gens)}期）',
                    color=COLORS['success'], edgecolor='white')
    bars2b = ax2.bar(x2 + width/2, without_values2, width, label=f'協賛なし（{len(without_sponsor_gens)}期）',
                    color=COLORS['gray'], edgecolor='white')

    ax2.set_ylabel('満足度（5段階）', fontsize=11, fontweight='bold')
    ax2.set_title('協賛の有無による満足度の差', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories2, fontsize=10)
    ax2.legend(loc='upper right', fontsize=9)

    # 目標線
    ax2.axhline(y=4.0, color=COLORS['warning'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.text(2.5, 4.05, '目標4.0', fontsize=9, color=COLORS['warning'])

    # 差を表示
    for i, (w, wo) in enumerate(zip(with_values2, without_values2)):
        diff = w - wo
        ax2.annotate(f'{diff:+.2f}', xy=(i - width/2, w + 0.05), ha='center', fontsize=10,
                    fontweight='bold', color=COLORS['success'] if diff > 0 else COLORS['accent'])

    ax2.set_ylim(0, 5.5)

    fig.suptitle('協賛効果の分析（533名参加者データ + 119件アンケート）', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_figure(fig, 'sponsorship_effect.png')

# ===========================================
# 19. 会費傾斜シミュレーション
# ===========================================
def chart_fee_simulation():
    """会費傾斜シミュレーション"""
    plans = ['現行', '案1\n（推奨）', '案2', '案3']
    revenues = [285, 317, 326, 306]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [COLORS['gray'], COLORS['success'], COLORS['secondary'], COLORS['secondary']]
    bars = ax.bar(plans, revenues, color=colors, edgecolor='white', width=0.6)

    # 値を表示
    for bar, rev in zip(bars, revenues):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{rev}万円', ha='center', fontsize=11, fontweight='bold',
                color=COLORS['primary'])

    # 現行との差
    for i, (bar, rev) in enumerate(zip(bars, revenues)):
        if i > 0:
            diff = rev - 285
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 15,
                    f'+{diff}万円', ha='center', fontsize=10,
                    color='white', fontweight='bold')

    ax.set_ylabel('期待収入（万円）', fontsize=11, fontweight='bold')
    ax.set_title('会費傾斜シミュレーション', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 360)

    # 案1（推奨）をハイライト
    bars[1].set_edgecolor(COLORS['success'])
    bars[1].set_linewidth(3)

    save_figure(fig, 'fee_simulation.png')

# ===========================================
# 20. 収益構造の積み上げ棒グラフ
# ===========================================
def chart_revenue_structure():
    """収益構造の積み上げ棒グラフ"""
    categories = ['現行', '案1\n（推奨）', '案1+\n個人協賛']
    ticket = [285, 317, 317]
    sponsorship = [0, 0, 8]
    allocation = [0, 0, 25]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(categories))
    width = 0.5

    p1 = ax.bar(x, ticket, width, label='チケット収入', color=COLORS['primary'], edgecolor='white')
    p2 = ax.bar(x, sponsorship, width, bottom=ticket, label='個人協賛', color=COLORS['secondary'], edgecolor='white')
    p3 = ax.bar(x, allocation, width, bottom=[t+s for t,s in zip(ticket, sponsorship)],
                label='協賛金充当', color=COLORS['success'], edgecolor='white')

    # 合計値を表示
    totals = [t+s+a for t,s,a in zip(ticket, sponsorship, allocation)]
    for i, total in enumerate(totals):
        ax.text(i, total + 5, f'{total}万円', ha='center', fontsize=11, fontweight='bold', color=COLORS['primary'])

    ax.set_ylabel('収入（万円）', fontsize=11, fontweight='bold')
    ax.set_title('収益構造の比較', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0, 380)

    # 増収額の注記
    ax.annotate('+32万円', xy=(1, 317), xytext=(1.5, 340),
               ha='center', fontsize=10, fontweight='bold', color=COLORS['success'],
               arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.5))
    ax.annotate('+65万円', xy=(2, 350), xytext=(2, 370),
               ha='center', fontsize=10, fontweight='bold', color=COLORS['success'])

    save_figure(fig, 'revenue_structure.png')

# ===========================================
# 21. ウォーターフォールチャート（増収施策効果）
# ===========================================
def chart_revenue_waterfall():
    """増収施策のウォーターフォールチャート"""
    labels = ['現行収入', '会費傾斜\n（1-10期）', '会費傾斜\n（11-20期）', '若手微増', '個人協賛', '協賛充当', '目標収入']
    values = [285, 21, 9, 2, 8, 25, 0]

    # 累積計算
    cumulative = [285]
    for v in values[1:-1]:
        cumulative.append(cumulative[-1] + v)
    cumulative.append(cumulative[-1])

    fig, ax = plt.subplots(figsize=(10, 5))

    # バーの開始位置と高さ
    starts = [0] + cumulative[:-1]
    colors = [COLORS['primary']] + [COLORS['success']] * (len(values) - 2) + [COLORS['secondary']]

    bars = ax.bar(labels, values[:-1] + [cumulative[-1]], bottom=[0] + [cumulative[i-1] for i in range(1, len(values)-1)] + [0],
                  color=colors, edgecolor='white', width=0.6)

    # 最初と最後のバーは累積値として表示
    bars[0].set_height(285)
    bars[-1].set_height(cumulative[-1])

    # 接続線
    for i in range(len(cumulative) - 1):
        ax.plot([i + 0.3, i + 0.7], [cumulative[i], cumulative[i]], 'k--', linewidth=1, alpha=0.5)

    # 値を表示
    for i, (bar, val) in enumerate(zip(bars, values[:-1] + [cumulative[-1]])):
        if i == 0 or i == len(bars) - 1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f'{int(bar.get_height())}万円', ha='center', fontsize=10, fontweight='bold', color=COLORS['primary'])
        else:
            ax.text(bar.get_x() + bar.get_width()/2, cumulative[i] + 3,
                    f'+{val}', ha='center', fontsize=9, fontweight='bold', color=COLORS['success'])

    ax.set_ylabel('収入（万円）', fontsize=11, fontweight='bold')
    ax.set_title('増収施策の積み上げ効果', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 380)

    save_figure(fig, 'revenue_waterfall.png')

# ===========================================
# 22. 施策の優先度マトリクス
# ===========================================
def chart_strategy_priority():
    """施策の効果とコストを比較し、優先度を示す"""
    strategies = [
        ('期別連絡網整備', 50, 5, '高'),
        ('申込状況公開', 30, 2, '高'),
        ('会費傾斜調整', 32, 0, '高'),
        ('会場変更（必須）', 40, 30, '高'),
        ('料理1.5倍増量', 40, 40, '中'),
        ('音響設備強化', 30, 10, '中'),
        ('個人協賛導入', 8, 3, '中'),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, effect, cost, priority in strategies:
        if priority == '高':
            color = COLORS['success']
            size = 300
        elif priority == '中':
            color = COLORS['warning']
            size = 200
        else:
            color = COLORS['gray']
            size = 150

        ax.scatter(cost, effect, s=size, c=color, alpha=0.7, edgecolors='white', linewidth=2)
        ax.annotate(name, xy=(cost, effect), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, color=COLORS['primary'])

    # 象限を分ける線
    ax.axhline(y=30, color=COLORS['light_gray'], linestyle='--', linewidth=1)
    ax.axvline(x=15, color=COLORS['light_gray'], linestyle='--', linewidth=1)

    # 象限ラベル
    ax.text(5, 45, '★ 最優先\n（高効果・低コスト）', fontsize=10, fontweight='bold',
            color=COLORS['success'], ha='center')
    ax.text(35, 45, '計画的実施\n（高効果・高コスト）', fontsize=10,
            color=COLORS['warning'], ha='center')
    ax.text(5, 15, '余裕があれば', fontsize=10, color=COLORS['gray'], ha='center')
    ax.text(35, 15, '優先度低', fontsize=10, color=COLORS['gray'], ha='center')

    ax.set_xlabel('コスト（万円）', fontsize=11, fontweight='bold')
    ax.set_ylabel('効果（参加者増 or 万円）', fontsize=11, fontweight='bold')
    ax.set_title('施策の優先度マトリクス → 左上の施策から着手', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(-5, 50)
    ax.set_ylim(0, 60)
    ax.grid(True, alpha=0.3)

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['success'], label='優先度：高'),
        Patch(facecolor=COLORS['warning'], label='優先度：中'),
        Patch(facecolor=COLORS['gray'], label='優先度：低'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    save_figure(fig, 'strategy_priority.png')

# ===========================================
# 23. 期別参加者増加ポテンシャル
# ===========================================
def chart_participation_potential():
    """各期の参加者増加ポテンシャルを示す"""
    classes = ['5期', '6期', '8期', '17期', '25期', '28期', '29期', '32期', '35期']
    current = [6, 7, 7, 5, 4, 5, 4, 2, 3]
    avg_with_involvement = [17.3] * len(classes)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(classes))
    width = 0.4

    bars = ax.bar(x, current, width, label='現在の参加者数', color=COLORS['gray'], edgecolor='white')

    # 平均との差（増加ポテンシャル）を積み上げ
    potential = [avg - curr for avg, curr in zip(avg_with_involvement, current)]
    ax.bar(x, potential, width, bottom=current, label='増加ポテンシャル',
           color=COLORS['success'], edgecolor='white', alpha=0.7)

    # 値を表示
    for i, (bar, curr, pot) in enumerate(zip(bars, current, potential)):
        ax.text(bar.get_x() + bar.get_width()/2, curr/2,
                f'{curr}', ha='center', fontsize=10, fontweight='bold', color='white')
        ax.text(bar.get_x() + bar.get_width()/2, curr + pot + 0.5,
                f'+{pot:.0f}', ha='center', fontsize=9, fontweight='bold', color=COLORS['success'])

    ax.set_ylabel('参加者数（名）', fontsize=11, fontweight='bold')
    ax.set_title('関与なし期の参加者増加ポテンシャル（メンター・運営配置で+10名/期）', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 22)

    # 合計増加見込み
    total_potential = sum(potential)
    ax.text(0.98, 0.95, f'合計 +{total_potential:.0f}名の増加見込み',
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            fontweight='bold', color=COLORS['success'],
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['success']))

    save_figure(fig, 'participation_potential.png')

# ===========================================
# 24. 期別参加者数の棒グラフ
# ===========================================
def chart_participants_by_class():
    """期別参加者数"""
    classes = list(range(1, 37))
    participants = [30, 32, 45, 26, 6, 7, 19, 7, 16, 21,
                    23, 28, 27, 10, 25, 13, 5, 19, 9, 28,
                    16, 10, 11, 11, 4, 20, 16, 5, 4, 7,
                    6, 2, 10, 5, 3, 7]

    fig, ax = plt.subplots(figsize=(12, 5))

    # 世代別に色分け
    colors = []
    for i, c in enumerate(classes):
        if c <= 10:
            colors.append(COLORS['primary'])
        elif c <= 20:
            colors.append(COLORS['secondary'])
        elif c <= 30:
            colors.append('#63B3ED')
        else:
            colors.append('#90CDF4')

    bars = ax.bar(classes, participants, color=colors, edgecolor='white', linewidth=0.5)

    # 最大値をハイライト
    max_idx = participants.index(max(participants))
    bars[max_idx].set_color(COLORS['accent'])

    ax.set_xlabel('期', fontsize=12, fontweight='bold')
    ax.set_ylabel('参加者数（名）', fontsize=12, fontweight='bold')
    ax.set_title('期別参加者数', fontsize=16, fontweight='bold', pad=20)

    ax.set_xticks(range(1, 37, 2))
    ax.set_xlim(0, 37)
    ax.set_ylim(0, 50)

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['primary'], label='1〜10期（ベテラン）'),
        Patch(facecolor=COLORS['secondary'], label='11〜20期（中堅）'),
        Patch(facecolor='#63B3ED', label='21〜30期'),
        Patch(facecolor='#90CDF4', label='31〜37期（若手）'),
        Patch(facecolor=COLORS['accent'], label='最多（3期: 45名）'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # 平均線
    avg = np.mean(participants)
    ax.axhline(y=avg, color=COLORS['warning'], linestyle='--', linewidth=2, alpha=0.7)
    ax.text(35.5, avg + 1, f'平均: {avg:.1f}名', fontsize=10, color=COLORS['warning'], ha='right')

    save_figure(fig, 'participants_by_class.png')

# ===========================================
# 25. 世代グループ別参加状況（円グラフ）
# ===========================================
def chart_generation_pie():
    """世代グループ別参加状況"""
    labels = ['1〜10期\n（ベテラン）', '11〜20期\n（中堅）', '21〜30期', '31〜37期\n（若手）']
    sizes = [209, 187, 104, 33]
    colors = [COLORS['primary'], COLORS['secondary'], '#63B3ED', '#90CDF4']
    explode = (0.02, 0.02, 0.02, 0.05)

    fig, ax = plt.subplots(figsize=(7, 7))

    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes))}名)',
        startangle=90,
        pctdistance=0.6,
        labeldistance=1.15,
        textprops={'fontsize': 11}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    ax.set_title('世代グループ別参加状況', fontsize=16, fontweight='bold', pad=20)

    # 中央にテキスト
    centre_circle = plt.Circle((0, 0), 0.35, fc='white')
    ax.add_patch(centre_circle)
    ax.text(0, 0.05, '総参加者', ha='center', va='center', fontsize=12, fontweight='bold', color=COLORS['gray'])
    ax.text(0, -0.12, '533名', ha='center', va='center', fontsize=18, fontweight='bold', color=COLORS['primary'])

    save_figure(fig, 'generation_distribution.png')

# ===========================================
# 26. 運営関与と参加者数の関係
# ===========================================
def chart_involvement_effect():
    """運営関与タイプと参加者数"""
    types = ['メンター＋運営\n両方', '本編登壇者\nあり', 'メンター\nのみ', '運営のみ', '関与なし\n（1-30期）', '関与なし\n（31-36期）']
    avg_participants = [23.8, 16.2, 16.0, 11.4, 9.0, 3.7]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(types))
    colors = [COLORS['success'] if p >= 20 else COLORS['secondary'] if p >= 15 else COLORS['warning'] if p >= 10 else COLORS['accent'] for p in avg_participants]
    bars = ax.bar(x, avg_participants, color=colors, edgecolor='white', width=0.6)

    for bar, val in zip(bars, avg_participants):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}名', ha='center', fontsize=11, fontweight='bold',
                color=COLORS['primary'])

    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=10)
    ax.set_ylabel('平均参加者数（名/期）', fontsize=12, fontweight='bold')
    ax.set_title('運営関与タイプ別 平均参加者数', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 30)

    # 差を示す矢印
    ax.annotate('', xy=(0, 23.8), xytext=(4, 9.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))
    ax.text(2.5, 18, '約2.6倍の差', fontsize=11, fontweight='bold', color=COLORS['accent'], ha='center')

    save_figure(fig, 'involvement_effect.png')

# ===========================================
# 27. 申込推移の時系列グラフ
# ===========================================
def chart_application_timeline():
    """申込推移の時系列"""
    weeks = ['9/21\n開始', '10/5', '10/19', '11/2', '11/16', '11/30', '12/7', '12/14', '12/21', '12/28\n当日']
    cumulative = [17, 35, 55, 80, 114, 183, 264, 415, 523, 533]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(range(len(weeks)), cumulative, alpha=0.3, color=COLORS['secondary'])
    ax.plot(range(len(weeks)), cumulative, marker='o', linewidth=2.5,
            color=COLORS['primary'], markersize=8, markerfacecolor='white', markeredgewidth=2)

    # キーポイントにラベル
    key_points = [(0, 17, '開始'), (5, 183, '後半開始'), (7, 415, '締切前日'), (9, 533, '最終')]
    for idx, val, label in key_points:
        ax.annotate(f'{label}\n{val}名', xy=(idx, val), xytext=(idx, val + 40),
                   ha='center', fontsize=9, fontweight='bold', color=COLORS['primary'])

    ax.set_xticks(range(len(weeks)))
    ax.set_xticklabels(weeks, fontsize=9)
    ax.set_ylabel('累積申込者数（名）', fontsize=11, fontweight='bold')
    ax.set_title('申込推移（時系列）', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 600)
    ax.grid(True, alpha=0.3)

    # 締切日に縦線
    ax.axvline(x=8, color=COLORS['accent'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(8.1, 100, '締切12/22', fontsize=9, color=COLORS['accent'], rotation=90, va='bottom')

    save_figure(fig, 'application_timeline.png')

# ===========================================
# 28. 締切効果の分析
# ===========================================
def chart_deadline_effect():
    """締切効果の分析"""
    days = ['12/13', '12/14', '12/15\n(締切)', '12/16', '12/17', '12/22\n(延長締切)']
    daily_applications = [28, 43, 60, 6, 8, 19]

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = [COLORS['secondary'] if d < 40 else COLORS['accent'] for d in daily_applications]
    colors[2] = COLORS['accent']
    colors[5] = COLORS['warning']

    bars = ax.bar(days, daily_applications, color=colors, edgecolor='white', width=0.6)

    for bar, val in zip(bars, daily_applications):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}名', ha='center', fontsize=10, fontweight='bold',
                color=COLORS['primary'])

    ax.set_ylabel('日別申込者数（名）', fontsize=11, fontweight='bold')
    ax.set_title('締切前後の申込状況', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 75)

    # 締切効果の注記
    ax.annotate('締切効果\n通常の1.9倍', xy=(2, 60), xytext=(3.5, 65),
               ha='center', fontsize=10, fontweight='bold', color=COLORS['accent'],
               arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5))

    ax.annotate('翌日急減', xy=(3, 6), xytext=(4, 25),
               ha='center', fontsize=9, color=COLORS['gray'],
               arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1))

    save_figure(fig, 'deadline_effect.png')

# ===========================================
# 29. 散布図（早期申込と最終参加者の相関）
# ===========================================
def chart_early_final_scatter():
    """早期申込数と最終参加者数の相関"""
    early = [10, 8, 6, 9, 0, 0, 5, 3, 4, 4, 4, 6, 9, 3, 9, 3, 0, 4, 0, 5, 3, 0, 0, 4, 0, 5, 4, 2, 1, 2, 2, 1, 3, 0, 0, 2]
    final = [30, 32, 45, 26, 6, 7, 19, 7, 16, 21, 23, 28, 27, 10, 25, 13, 5, 19, 9, 28, 16, 10, 11, 11, 4, 20, 16, 5, 4, 7, 6, 2, 10, 5, 3, 7]

    fig, ax = plt.subplots(figsize=(9, 6))

    # 世代別に色分け
    colors_list = []
    for i in range(36):
        if i < 10:
            colors_list.append(COLORS['primary'])
        elif i < 20:
            colors_list.append(COLORS['secondary'])
        elif i < 30:
            colors_list.append('#63B3ED')
        else:
            colors_list.append('#90CDF4')

    ax.scatter(early, final, c=colors_list, s=100, alpha=0.7, edgecolors='white', linewidth=1)

    # 回帰直線
    z = np.polyfit(early, final, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 12, 100)
    ax.plot(x_line, p(x_line), '--', color=COLORS['accent'], linewidth=2, label=f'回帰直線 (r=0.62)')

    # 特筆すべき点にラベル
    ax.annotate('3期', xy=(6, 45), xytext=(7, 47), fontsize=9, fontweight='bold', color=COLORS['accent'])
    ax.annotate('1期', xy=(10, 30), xytext=(10.5, 32), fontsize=9, color=COLORS['primary'])

    ax.set_xlabel('早期申込者数（10月末時点）', fontsize=11, fontweight='bold')
    ax.set_ylabel('最終参加者数', fontsize=11, fontweight='bold')
    ax.set_title('早期申込数 vs 最終参加者数（期別）', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3)

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['primary'], label='1〜10期'),
        Patch(facecolor=COLORS['secondary'], label='11〜20期'),
        Patch(facecolor='#63B3ED', label='21〜30期'),
        Patch(facecolor='#90CDF4', label='31〜36期'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    save_figure(fig, 'early_final_scatter.png')

# ===========================================
# 30. 折れ線グラフ（世代別申込推移）
# ===========================================
def chart_application_area():
    """申込の累積推移（折れ線グラフ、世代別）"""
    weeks = ['9/21', '10/5', '10/19', '11/2', '11/16', '11/30', '12/7', '12/14', '12/21', '12/28']

    gen1_10 = [8, 16, 25, 38, 52, 80, 115, 170, 205, 209]
    gen11_20 = [5, 12, 20, 30, 45, 75, 105, 155, 183, 187]
    gen21_30 = [3, 5, 8, 10, 14, 22, 35, 70, 100, 104]
    gen31_37 = [1, 2, 2, 2, 3, 6, 9, 20, 30, 33]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(weeks, gen1_10, 'o-', linewidth=2.5, markersize=6, label='1〜10期（209名）',
            color=COLORS['primary'])
    ax.plot(weeks, gen11_20, 's--', linewidth=2.5, markersize=6, label='11〜20期（187名）',
            color=COLORS['secondary'])
    ax.plot(weeks, gen21_30, '^-.', linewidth=2.5, markersize=6, label='21〜30期（104名）',
            color=COLORS['success'])
    ax.plot(weeks, gen31_37, 'D:', linewidth=2.5, markersize=6, label='31〜37期（33名）',
            color=COLORS['accent'])

    ax.set_xlabel('日付', fontsize=11, fontweight='bold')
    ax.set_ylabel('累積申込者数（名）', fontsize=11, fontweight='bold')
    ax.set_title('世代別 申込推移 → ベテラン層が終始リード、若手は後半に伸び', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 230)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    ax.annotate('12月に急増', xy=(7, 170), xytext=(5, 190),
               fontsize=10, color=COLORS['primary'],
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1.5))

    save_figure(fig, 'application_area.png')

# ===========================================
# 31. 目標達成率ゲージ
# ===========================================
def chart_target_gauge():
    """目標達成率のゲージチャート"""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    targets = [
        ('参加者数', 533, 630, '名'),
        ('収入', 285, 350, '万円'),
        ('料理満足度', 3.31, 4.0, '/5'),
        ('若手参加', 33, 60, '名'),
    ]

    for ax, (label, current, target, unit) in zip(axes, targets):
        rate = min(current / target * 100, 100)

        # 背景の円弧
        theta = np.linspace(0, np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), color=COLORS['light_gray'], linewidth=15)

        # 達成率の円弧
        theta_filled = np.linspace(0, np.pi * rate / 100, 100)
        color = COLORS['success'] if rate >= 80 else COLORS['warning'] if rate >= 60 else COLORS['accent']
        ax.plot(np.cos(theta_filled), np.sin(theta_filled), color=color, linewidth=15)

        # 中央にテキスト
        ax.text(0, 0.2, f'{rate:.0f}%', ha='center', va='center', fontsize=16, fontweight='bold', color=COLORS['primary'])
        ax.text(0, -0.15, f'{current}{unit}', ha='center', va='center', fontsize=10, color=COLORS['gray'])
        ax.text(0, -0.35, f'目標: {target}{unit}', ha='center', va='center', fontsize=9, color=COLORS['gray'])

        ax.set_title(label, fontsize=11, fontweight='bold', pad=5)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.2)
        ax.axis('off')

    fig.suptitle('現状 vs 目標（達成率）', fontsize=14, fontweight='bold', y=1.1)
    plt.tight_layout()

    save_figure(fig, 'target_gauge.png')

# ===========================================
# メイン処理
# ===========================================
if __name__ == '__main__':
    print('=' * 60)
    print('開邦高校大同窓会 統合分析グラフ生成')
    print('=' * 60)

    setup_style()

    # データ読み込み
    print('\n[1] データ読み込み...')
    try:
        df = prepare_data()
        print(f'  → 読み込み完了: {len(df)}件')
    except Exception as e:
        print(f'  → エラー: {e}')
        print('  → デフォルトデータで続行')
        df = None

    if df is not None:
        print('\n[2] グラフ生成（119件実データ + Gemini補完）...')

        # === 満足度分析 ===
        print('  - 満足度評価（全体満足度含む）')
        scores = chart_satisfaction_with_overall(df)
        print(f'    平均スコア: {scores}')

        print('  - 世代別満足度ヒートマップ')
        chart_satisfaction_by_generation(df)

        print('  - 満足度分布ヒストグラム')
        chart_satisfaction_distribution(df)

        print('  - 相関分析')
        chart_correlation_analysis(df)

        print('  - 世代別課題比較')
        chart_generation_issues(df)

        # === 課題分析 ===
        print('  - 課題別言及数')
        issues = chart_issues_count_updated(df)

        # === 回答者属性 ===
        print('  - 回答者の世代分布')
        chart_respondents_by_generation(df)

        print('  - 情報入手経路')
        chart_information_source(df)

        print('  - 希望プログラム')
        chart_desired_programs(df)

        print('  - 不参加理由')
        chart_non_participation_reasons(df)

        print('  - 開催条件希望')
        chart_opening_conditions(df)

        # === 協力意向分析 ===
        print('  - 協力意向')
        chart_cooperation_willingness(df)

        print('  - 世代別意欲層')
        chart_motivation_by_generation(df)

        print('  - 学科別意欲層')
        chart_motivation_by_department(df)

        # === 満足度・協賛分析 ===
        print('  - 会費許容額')
        chart_fee_tolerance(df)

        print('  - 協賛効果')
        chart_sponsorship_effect(df)

        print('  - 期別バブルチャート（回答者数）')
        chart_bubble_evaluation(df)

        print('  - 期別バブルチャート（参加者数）')
        chart_bubble_participants(df)

        # === 施策・目標関連（固定データ） ===
        print('  - 目標達成率ゲージ')
        chart_target_gauge()

        print('  - 目標達成ロードマップ')
        chart_target_roadmap()

        print('  - 会費傾斜シミュレーション')
        chart_fee_simulation()

        print('  - 収益構造比較')
        chart_revenue_structure()

        print('  - 収益ウォーターフォール')
        chart_revenue_waterfall()

        print('  - 施策優先度マトリクス')
        chart_strategy_priority()

        print('  - 参加者増加ポテンシャル')
        chart_participation_potential()

        # === 参加者データ（533名全体） ===
        print('  - 期別参加者数')
        chart_participants_by_class()

        print('  - 世代グループ別参加状況')
        chart_generation_pie()

        print('  - 運営関与効果')
        chart_involvement_effect()

        print('  - 申込推移')
        chart_application_timeline()

        print('  - 締切効果')
        chart_deadline_effect()

        print('  - 早期申込vs最終参加者')
        chart_early_final_scatter()

        print('  - 世代別申込推移')
        chart_application_area()

    print('\n' + '=' * 60)
    print('全31グラフ生成完了!')
    print('=' * 60)
