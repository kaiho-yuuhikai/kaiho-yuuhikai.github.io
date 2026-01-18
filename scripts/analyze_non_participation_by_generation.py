#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
不参加理由の世代別分析
- 世代×不参加理由のクロス集計
- ヒートマップ可視化
- 統計的考察の自動生成
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Hiragino Sans'

# カラー設定
COLORS = {
    'primary': '#1A365D',
    'secondary': '#3182CE',
    'accent': '#E53E3E',
    'success': '#38A169',
    'gray': '#718096',
    'light_gray': '#E2E8F0',
    'background': '#F7FAFC'
}

def load_data():
    """データ読み込み"""
    filepath = '【卒業生・教職員用】第3回 開邦高校大同窓会 事後アンケート（回答） - フォームの回答 1 (3).tsv'
    df = pd.read_csv(filepath, sep='\t', encoding='utf-8')

    # 期の数値化
    ki_col = [c for c in df.columns if '何期' in c or '期生' in c]
    if ki_col:
        ki_series = df[ki_col[0]].astype(str)
        df['期_数値'] = pd.to_numeric(ki_series.str.extract(r'(\d+)')[0], errors='coerce')

    return df

def get_generation_group(ki):
    """期から世代グループを返す"""
    if pd.isna(ki):
        return None
    if ki <= 10:
        return '1〜10期\n(ベテラン)'
    elif ki <= 20:
        return '11〜20期\n(中堅)'
    elif ki <= 30:
        return '21〜30期'
    else:
        return '31期以降\n(若手)'

def analyze_non_participation(df):
    """不参加理由の世代別分析"""
    # 不参加理由の列を特定
    reason_col = [c for c in df.columns if '不参加' in c and '理由' in c]
    if not reason_col:
        print("不参加理由の列が見つかりません")
        return None

    # 世代グループを追加
    df['世代グループ'] = df['期_数値'].apply(get_generation_group)

    # 不参加理由のカテゴリ
    reasons_keywords = {
        '仕事の都合': ['仕事', '業務', '勤務'],
        '県外・海外在住': ['県外', '海外', '在住', '帰省'],
        '広報不足': ['知らなかった', '情報', '広報'],
        '土曜日希望': ['土曜', '日曜'],
        '家庭の事情': ['家庭', '子育て', '介護', '育児'],
        '会費が高い': ['会費', '高い', '料金'],
    }

    generations = ['1〜10期\n(ベテラン)', '11〜20期\n(中堅)', '21〜30期', '31期以降\n(若手)']

    # クロス集計用のデータフレーム
    cross_data = pd.DataFrame(index=reasons_keywords.keys(), columns=generations)
    cross_counts = pd.DataFrame(index=reasons_keywords.keys(), columns=generations)
    gen_totals = {}

    for gen in generations:
        gen_df = df[df['世代グループ'] == gen]
        gen_totals[gen] = len(gen_df)
        reason_data = gen_df[reason_col[0]].dropna()

        for reason, keywords in reasons_keywords.items():
            count = 0
            for text in reason_data:
                for kw in keywords:
                    if kw in str(text):
                        count += 1
                        break
            cross_counts.loc[reason, gen] = count
            # 各世代の回答者数に対する割合
            cross_data.loc[reason, gen] = count / len(gen_df) * 100 if len(gen_df) > 0 else 0

    return cross_data.astype(float), cross_counts.astype(int), gen_totals

def create_heatmap(cross_data, cross_counts, gen_totals):
    """ヒートマップを作成"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # データを配列に変換
    data = cross_data.values

    # ヒートマップを描画
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=50)

    # 軸ラベルの設定
    generations = cross_data.columns.tolist()
    reasons = cross_data.index.tolist()

    ax.set_xticks(np.arange(len(generations)))
    ax.set_yticks(np.arange(len(reasons)))
    ax.set_xticklabels([f"{g}\n(n={gen_totals[g]})" for g in generations], fontsize=11, fontweight='bold')
    ax.set_yticklabels(reasons, fontsize=11)

    # 各セルに値を表示
    for i in range(len(reasons)):
        for j in range(len(generations)):
            value = data[i, j]
            count = cross_counts.iloc[i, j]
            text_color = 'white' if value > 25 else 'black'
            ax.text(j, i, f'{value:.1f}%\n({count}件)', ha='center', va='center',
                   fontsize=10, fontweight='bold', color=text_color)

    # カラーバー
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('言及率（%）', fontsize=11, fontweight='bold')

    # タイトル
    ax.set_title('世代別 不参加理由の傾向（ヒートマップ）', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('世代グループ', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../images/analysis/non_participation_heatmap.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: images/analysis/non_participation_heatmap.png")

    return True

def generate_analysis(cross_data, cross_counts, gen_totals):
    """考察を自動生成"""
    generations = cross_data.columns.tolist()
    reasons = cross_data.index.tolist()

    analysis = []
    analysis.append("### 世代別 不参加理由の分析\n")

    # 各世代の特徴を分析
    analysis.append("#### 世代別の特徴\n")

    for gen in generations:
        gen_data = cross_data[gen].sort_values(ascending=False)
        top_reason = gen_data.index[0]
        top_value = gen_data.iloc[0]
        analysis.append(f"**{gen.replace(chr(10), ' ')}** (n={gen_totals[gen]})")
        analysis.append(f"- 最も多い理由: **{top_reason}** ({top_value:.1f}%)")

        # 全体平均との比較
        avg = cross_data[gen].mean()
        analysis.append(f"- 平均言及率: {avg:.1f}%\n")

    # 理由別の世代差分析
    analysis.append("#### 不参加理由別の世代間差異\n")

    for reason in reasons:
        reason_data = cross_data.loc[reason]
        max_gen = reason_data.idxmax()
        max_val = reason_data.max()
        min_gen = reason_data.idxmin()
        min_val = reason_data.min()
        diff = max_val - min_val

        if diff > 10:  # 10%以上の差がある場合に言及
            analysis.append(f"**{reason}**")
            analysis.append(f"- 世代間差: {diff:.1f}ポイント")
            analysis.append(f"- 最多: {max_gen.replace(chr(10), ' ')} ({max_val:.1f}%)")
            analysis.append(f"- 最少: {min_gen.replace(chr(10), ' ')} ({min_val:.1f}%)\n")

    # 主要な発見
    analysis.append("#### 主要な発見\n")

    # 1. 若手の特徴
    young_data = cross_data['31期以降\n(若手)']
    young_top = young_data.idxmax()
    analysis.append(f"1. **若手層（31期以降）の特徴**: {young_top}が最も多い（{young_data[young_top]:.1f}%）")

    # 2. ベテランの特徴
    veteran_data = cross_data['1〜10期\n(ベテラン)']
    veteran_top = veteran_data.idxmax()
    analysis.append(f"2. **ベテラン層（1〜10期）の特徴**: {veteran_top}が最も多い（{veteran_data[veteran_top]:.1f}%）")

    # 3. 広報不足の世代差
    pr_data = cross_data.loc['広報不足']
    analysis.append(f"3. **広報不足の世代差**: {pr_data.idxmax().replace(chr(10), ' ')}で最も高い（{pr_data.max():.1f}%）→ 若手への広報強化が必要")

    # 4. 会費の影響
    fee_data = cross_data.loc['会費が高い']
    analysis.append(f"4. **会費の影響**: 全体的に低い（最大{fee_data.max():.1f}%）→ 会費は主要な障壁ではない")

    return '\n'.join(analysis)

def main():
    print("=" * 60)
    print("不参加理由の世代別分析")
    print("=" * 60)

    # データ読み込み
    df = load_data()
    print(f"データ読み込み完了: {len(df)}件")

    # 分析実行
    cross_data, cross_counts, gen_totals = analyze_non_participation(df)

    if cross_data is not None:
        print("\n【クロス集計結果（%）】")
        print(cross_data.round(1).to_string())

        print("\n【実数】")
        print(cross_counts.to_string())

        # ヒートマップ作成
        create_heatmap(cross_data, cross_counts, gen_totals)

        # 考察生成
        analysis = generate_analysis(cross_data, cross_counts, gen_totals)
        print("\n" + "=" * 60)
        print("【自動生成された考察】")
        print("=" * 60)
        print(analysis)

        # 考察をファイルに保存
        with open('../reports/non_participation_analysis.md', 'w', encoding='utf-8') as f:
            f.write("# 不参加理由の世代別分析\n\n")
            f.write(f"<img src=\"../images/analysis/non_participation_heatmap.png\" alt=\"世代別不参加理由ヒートマップ\" style=\"max-width: 100%; width: 800px;\">\n\n")
            f.write(analysis)
        print("\nSaved: reports/non_participation_analysis.md")

if __name__ == '__main__':
    main()
