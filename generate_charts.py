#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3回 開邦高校大同窓会 データ分析レポート用グラフ生成
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# 日本語フォント設定
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# カラーパレット
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
    fig.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {filepath}')

# ===========================================
# 1. 期別参加者数の棒グラフ
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
# 2. 世代グループ別参加状況（円グラフ）
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
# 3. 満足度評価（横棒グラフ）
# ===========================================
def chart_satisfaction():
    """満足度評価"""
    categories = ['時間帯\n(15:00-17:30)', '日程\n(12/28)', '会費妥当性', '料理・ドリンク']
    scores = [4.29, 4.05, 3.90, 3.34]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # 色を満足度に応じて変更
    colors = []
    for s in scores:
        if s >= 4.0:
            colors.append(COLORS['success'])
        elif s >= 3.5:
            colors.append(COLORS['warning'])
        else:
            colors.append(COLORS['accent'])

    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, scores, color=colors, height=0.6, edgecolor='white')

    # スコアをバーの右に表示
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.1, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}', va='center', fontsize=12, fontweight='bold',
                color=colors[i])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlim(0, 5.5)
    ax.set_xlabel('満足度スコア（5段階評価）', fontsize=12, fontweight='bold')
    ax.set_title('満足度評価（項目別）', fontsize=16, fontweight='bold', pad=20)

    # 目標線
    ax.axvline(x=4.0, color=COLORS['gray'], linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(4.05, 3.7, '目標: 4.0', fontsize=9, color=COLORS['gray'])

    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['success'], label='良好（4.0以上）'),
        Patch(facecolor=COLORS['warning'], label='普通（3.5〜4.0）'),
        Patch(facecolor=COLORS['accent'], label='要改善（3.5未満）'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    ax.invert_yaxis()

    save_figure(fig, 'satisfaction_scores.png')

# ===========================================
# 4. 情報入手経路（円グラフ）
# ===========================================
def chart_information_source():
    """情報入手経路"""
    labels = ['同窓生からの\n口コミ', '雄飛会\nFacebook', '大同窓会\nInstagram', 'ポスター', 'その他']
    sizes = [83.1, 23.7, 13.6, 5.1, 5.0]
    colors = [COLORS['primary'], COLORS['secondary'], '#63B3ED', '#90CDF4', COLORS['light_gray']]

    fig, ax = plt.subplots(figsize=(7, 6))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.75,
        labeldistance=1.15,
        textprops={'fontsize': 10}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('情報入手経路（複数回答）', fontsize=16, fontweight='bold', pad=20)

    save_figure(fig, 'information_source.png')

# ===========================================
# 5. 不参加理由（横棒グラフ）
# ===========================================
def chart_non_participation_reasons():
    """不参加理由"""
    reasons = ['仕事の都合', '県外・海外在住', '広報不足', '土曜日希望', '家庭の事情', '会費が高い']
    percentages = [39.0, 35.6, 20.3, 8.5, 15.0, 6.8]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    y_pos = np.arange(len(reasons))
    bars = ax.barh(y_pos, percentages, color=COLORS['secondary'], height=0.6, edgecolor='white')

    # 最も多い理由をハイライト
    bars[0].set_color(COLORS['primary'])
    bars[1].set_color(COLORS['primary'])

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
# 6. 希望プログラム（横棒グラフ）
# ===========================================
def chart_desired_programs():
    """希望プログラム"""
    programs = [
        '校歌斉唱\n（芸術科合唱つき）',
        '思い出ビデオ\n・スライドショー',
        '学科・専門分野別\n交流コーナー',
        '在校生の\n活動紹介',
        '卒業生有志の\n音楽・パフォーマンス',
        'スマホ参加型\n企画（クイズ等）'
    ]
    percentages = [49.2, 47.5, 42.4, 40.7, 35.6, 32.2]

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
    ax.set_xlim(0, 60)
    ax.set_xlabel('希望率（%）', fontsize=12, fontweight='bold')
    ax.set_title('次回取り入れてほしいプログラム TOP6', fontsize=16, fontweight='bold', pad=20)

    ax.invert_yaxis()

    save_figure(fig, 'desired_programs.png')

# ===========================================
# 7. 会費許容額の分布
# ===========================================
def chart_fee_tolerance():
    """会費許容額の世代別分布"""
    generations = ['1〜10期\n（ベテラン）', '11〜20期\n（中堅）', '21〜30期', '31〜37期\n（若手）']
    current_fee = [5500, 5500, 5500, 3000]
    tolerance = [6176, 6167, 5800, 5000]

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
# 8. 協力意欲の高い層の分析
# ===========================================
def chart_cooperation_willingness():
    """協力内容の傾向"""
    items = [
        '寄付や協賛\nによる支援',
        '特別授業等の\n講師',
        '具体的な相談が\nあれば検討',
        '次回の\n実行委員',
        '広報協力\n（SNS等）',
        '雄飛会の\n役員活動'
    ]
    counts = [17, 14, 10, 6, 6, 3]
    percentages = [63.0, 51.9, 37.0, 22.2, 22.2, 11.1]

    fig, ax = plt.subplots(figsize=(9, 5))

    y_pos = np.arange(len(items))
    colors = [COLORS['primary'] if p >= 50 else COLORS['secondary'] if p >= 30 else '#63B3ED' for p in percentages]
    bars = ax.barh(y_pos, percentages, color=colors, height=0.6, edgecolor='white')

    for bar, pct, cnt in zip(bars, percentages, counts):
        ax.text(pct + 2, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%（{cnt}名）', va='center', fontsize=11, fontweight='bold',
                color=COLORS['primary'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(items, fontsize=10)
    ax.set_xlim(0, 80)
    ax.set_xlabel('割合（%）/ 意欲の高い27名中', fontsize=12, fontweight='bold')
    ax.set_title('協力内容の傾向（取組意欲の高い層）', fontsize=16, fontweight='bold', pad=20)

    ax.invert_yaxis()

    save_figure(fig, 'cooperation_willingness.png')

# ===========================================
# 9. 運営関与と参加者数の関係
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
# 10. 申込推移の時系列グラフ
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
# 11. 協賛効果の比較
# ===========================================
def chart_sponsorship_effect():
    """協賛あり vs なしの比較"""
    categories = ['平均参加者数\n（名/期）', '料理満足度\n（/5.0）', '会費満足度\n（/5.0）']
    with_sponsor = [21.2, 3.50, 4.13]
    without_sponsor = [10.2, 3.05, 3.48]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, with_sponsor, width, label='協賛あり（15期）',
                   color=COLORS['success'], edgecolor='white')
    bars2 = ax.bar(x + width/2, without_sponsor, width, label='協賛なし（21期）',
                   color=COLORS['gray'], edgecolor='white')

    ax.set_ylabel('値', fontsize=11, fontweight='bold')
    ax.set_title('協賛の有無による参加者数・満足度の差', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(loc='upper right', fontsize=9)

    # 差を表示
    for i, (w, wo) in enumerate(zip(with_sponsor, without_sponsor)):
        diff = w - wo
        if i == 0:
            label = f'+{diff:.1f}名\n(+{(diff/wo)*100:.0f}%)'
        else:
            label = f'+{diff:.2f}'
        ax.annotate(label, xy=(i, max(w, wo) + 0.3), ha='center', fontsize=10,
                   fontweight='bold', color=COLORS['success'])

    ax.set_ylim(0, 26)

    save_figure(fig, 'sponsorship_effect.png')

# ===========================================
# 12. 傾斜シミュレーション
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
# 13. 世代別意欲層の分布
# ===========================================
def chart_motivation_by_generation():
    """世代別の取組意欲の高い層"""
    generations = ['1〜10期\n（ベテラン）', '11〜20期\n（中堅）', '21〜30期', '31期以降\n（新世代）']
    motivated = [11, 10, 4, 2]
    percentages = [40.7, 37.0, 14.8, 7.4]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [COLORS['primary'], COLORS['secondary'], '#63B3ED', '#90CDF4']
    bars = ax.bar(generations, motivated, color=colors, edgecolor='white', width=0.6)

    for bar, val, pct in zip(bars, motivated, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val}名\n({pct}%)', ha='center', fontsize=10, fontweight='bold',
                color=COLORS['primary'])

    ax.set_ylabel('人数（名）', fontsize=11, fontweight='bold')
    ax.set_title('世代別 取組意欲の高い層（連絡可回答者）', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 15)

    # 78%がベテラン・中堅という注記
    ax.annotate('ベテラン・中堅で\n78%を占める', xy=(0.5, 11), xytext=(2.5, 13),
               ha='center', fontsize=10, fontweight='bold', color=COLORS['accent'],
               arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5))

    save_figure(fig, 'motivation_by_generation.png')

# ===========================================
# 14. 学科別意欲層の分布
# ===========================================
def chart_motivation_by_department():
    """学科別の取組意欲"""
    departments = ['理数科', '芸術科', '英語科', '学術探究科']
    total = [36, 13, 8, 1]
    motivated = [18, 5, 3, 1]
    rates = [50.0, 38.5, 37.5, 100.0]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(departments))
    width = 0.35

    bars1 = ax.bar(x - width/2, total, width, label='全回答者',
                   color=COLORS['light_gray'], edgecolor='white')
    bars2 = ax.bar(x + width/2, motivated, width, label='意欲高い層',
                   color=COLORS['secondary'], edgecolor='white')

    # 割合を表示
    for i, (t, m, r) in enumerate(zip(total, motivated, rates)):
        ax.text(i + width/2, m + 0.5, f'{r:.1f}%', ha='center', fontsize=10,
               fontweight='bold', color=COLORS['primary'])

    ax.set_ylabel('人数（名）', fontsize=11, fontweight='bold')
    ax.set_title('学科別 取組意欲の割合', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(departments, fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 45)

    # 理数科が最多という注記
    ax.annotate('理数科が最も\n協力意欲が高い', xy=(0, 18), xytext=(1.5, 30),
               ha='center', fontsize=10, fontweight='bold', color=COLORS['success'],
               arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.5))

    save_figure(fig, 'motivation_by_department.png')

# ===========================================
# 15. 締切効果の分析
# ===========================================
def chart_deadline_effect():
    """締切効果の分析"""
    days = ['12/13', '12/14', '12/15\n(締切)', '12/16', '12/17', '12/22\n(延長締切)']
    daily_applications = [28, 43, 60, 6, 8, 19]

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = [COLORS['secondary'] if d < 40 else COLORS['accent'] for d in daily_applications]
    colors[2] = COLORS['accent']  # 締切日
    colors[5] = COLORS['warning']  # 延長締切日

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
# 16. 課題別言及数
# ===========================================
def chart_issues_count():
    """自由回答から抽出した課題"""
    issues = ['料理の不足', '音響問題', '世代間交流', '視認性', '会場の狭さ']
    counts = [13, 13, 13, 11, 9]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    colors = [COLORS['accent'] if c >= 13 else COLORS['warning'] for c in counts]
    y_pos = np.arange(len(issues))
    bars = ax.barh(y_pos, counts, color=colors, height=0.6, edgecolor='white')

    for bar, cnt in zip(bars, counts):
        ax.text(cnt + 0.3, bar.get_y() + bar.get_height()/2,
                f'{cnt}回', va='center', fontsize=11, fontweight='bold',
                color=COLORS['primary'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(issues, fontsize=10)
    ax.set_xlim(0, 18)
    ax.set_xlabel('言及数（回）', fontsize=11, fontweight='bold')
    ax.set_title('自由回答から抽出した課題 TOP5', fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()

    save_figure(fig, 'issues_count.png')

# ===========================================
# 17. レーダーチャート（世代別特性比較）
# ===========================================
def chart_generation_radar():
    """世代別の特性をレーダーチャートで比較"""
    from math import pi

    categories = ['参加率', '会費満足度', '許容額余地', '内容重視', '協力意欲', '口コミ到達']
    N = len(categories)

    # 各世代のデータ（0-100にスケール）
    data = {
        '1〜10期': [85, 85, 80, 100, 90, 85],
        '11〜20期': [75, 76, 78, 50, 80, 80],
        '21〜30期': [50, 74, 35, 0, 45, 75],
        '31〜37期': [30, 55, 100, 0, 25, 70],
    }

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = [COLORS['primary'], COLORS['secondary'], '#63B3ED', '#90CDF4']
    for (gen, values), color in zip(data.items(), colors):
        values = values + values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=gen, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title('世代別 特性比較（レーダーチャート）', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)

    save_figure(fig, 'generation_radar.png')

# ===========================================
# 18. 散布図（早期申込と最終参加者の相関）
# ===========================================
def chart_early_final_scatter():
    """早期申込数と最終参加者数の相関"""
    # 各期のデータ（早期申込数, 最終参加者数）
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
# 19. 積み上げ棒グラフ（収益構造）
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
# 20. ヒートマップ（世代×満足度項目）
# ===========================================
def chart_satisfaction_heatmap():
    """世代別満足度のヒートマップ"""
    generations = ['1〜10期', '11〜20期', '21〜30期', '31〜37期']
    items = ['時間帯', '日程', '会費', '料理']

    # 満足度データ (4世代 x 4項目)
    data = np.array([
        [4.5, 4.2, 4.25, 3.6],
        [4.3, 4.1, 3.81, 3.3],
        [4.2, 3.9, 3.70, 3.2],
        [4.0, 3.8, 2.75, 3.0],
    ])

    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=2.5, vmax=5.0)

    ax.set_xticks(np.arange(len(items)))
    ax.set_yticks(np.arange(len(generations)))
    ax.set_xticklabels(items, fontsize=10)
    ax.set_yticklabels(generations, fontsize=10)

    # 各セルに値を表示
    for i in range(len(generations)):
        for j in range(len(items)):
            color = 'white' if data[i, j] < 3.5 else 'black'
            ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                   fontsize=11, fontweight='bold', color=color)

    ax.set_title('世代別 満足度ヒートマップ', fontsize=14, fontweight='bold', pad=15)

    # カラーバー
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('満足度（5段階）', rotation=-90, va='bottom', fontsize=10)

    save_figure(fig, 'satisfaction_heatmap.png')

# ===========================================
# 21. ウォーターフォールチャート（増収施策効果）
# ===========================================
def chart_revenue_waterfall():
    """増収施策のウォーターフォールチャート"""
    labels = ['現行収入', '会費傾斜\n（1-10期）', '会費傾斜\n（11-20期）', '若手微増', '個人協賛', '協賛充当', '目標収入']
    values = [285, 21, 9, 2, 8, 25, 0]  # 最後は計算で埋める

    # 累積計算
    cumulative = [285]
    for v in values[1:-1]:
        cumulative.append(cumulative[-1] + v)
    cumulative.append(cumulative[-1])  # 最終値

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
# 22. ドーナツチャート（開催希望条件）
# ===========================================
def chart_opening_conditions():
    """開催希望条件のドーナツチャート"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 開催頻度
    freq_labels = ['5年に1回', '3年に1回', '毎年', 'その他']
    freq_sizes = [52.5, 30.5, 10.2, 6.8]
    axes[0].pie(freq_sizes, labels=freq_labels, autopct='%1.1f%%', startangle=90,
                colors=[COLORS['primary'], COLORS['secondary'], '#63B3ED', COLORS['light_gray']],
                wedgeprops=dict(width=0.5, edgecolor='white'), textprops={'fontsize': 9})
    axes[0].set_title('開催頻度', fontsize=12, fontweight='bold')

    # 開催時期
    time_labels = ['年末年始', '夏休み', 'GW', 'その他']
    time_sizes = [81.4, 10.2, 5.1, 3.3]
    axes[1].pie(time_sizes, labels=time_labels, autopct='%1.1f%%', startangle=90,
                colors=[COLORS['primary'], COLORS['secondary'], '#63B3ED', COLORS['light_gray']],
                wedgeprops=dict(width=0.5, edgecolor='white'), textprops={'fontsize': 9})
    axes[1].set_title('開催時期', fontsize=12, fontweight='bold')

    # 曜日
    day_labels = ['土曜日', '日曜日', '平日', '不問']
    day_sizes = [33.9, 28.8, 5.1, 32.2]
    axes[2].pie(day_sizes, labels=day_labels, autopct='%1.1f%%', startangle=90,
                colors=[COLORS['primary'], COLORS['secondary'], '#63B3ED', COLORS['light_gray']],
                wedgeprops=dict(width=0.5, edgecolor='white'), textprops={'fontsize': 9})
    axes[2].set_title('希望曜日', fontsize=12, fontweight='bold')

    fig.suptitle('開催希望条件', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()

    save_figure(fig, 'opening_conditions.png')

# ===========================================
# 23. バブルチャート（期別の総合評価）
# ===========================================
def chart_bubble_evaluation():
    """期別の総合評価（バブルチャート）"""
    # データ: (参加者数, 満足度, 協賛額/万円)
    data = {
        '3期': (45, 4.5, 12),
        '1期': (30, 4.0, 10),
        '2期': (32, 3.6, 1),
        '13期': (27, 3.8, 20),
        '4期': (26, 3.8, 13.7),
        '20期': (28, 3.6, 9.7),
        '12期': (28, 3.7, 0),
        '15期': (25, 3.6, 4),
        '11期': (23, 2.8, 0),
        '10期': (21, 4.0, 10),
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, (participants, satisfaction, sponsorship) in data.items():
        size = sponsorship * 30 + 50  # バブルサイズ
        color = COLORS['primary'] if int(name.replace('期', '')) <= 10 else COLORS['secondary']
        ax.scatter(participants, satisfaction, s=size, alpha=0.6, color=color, edgecolors='white', linewidth=2)
        ax.annotate(name, xy=(participants, satisfaction), xytext=(5, 5),
                   textcoords='offset points', fontsize=9, color=COLORS['primary'])

    ax.set_xlabel('参加者数（名）', fontsize=11, fontweight='bold')
    ax.set_ylabel('平均満足度', fontsize=11, fontweight='bold')
    ax.set_title('期別 参加者数 × 満足度 × 協賛額', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(15, 50)
    ax.set_ylim(2.5, 5.0)
    ax.grid(True, alpha=0.3)

    # 凡例（バブルサイズ）
    for size, label in [(50, '0万円'), (200, '5万円'), (350, '10万円')]:
        ax.scatter([], [], s=size, c=COLORS['gray'], alpha=0.5, label=label, edgecolors='white')
    ax.legend(title='協賛額', loc='lower right', fontsize=9)

    save_figure(fig, 'bubble_evaluation.png')

# ===========================================
# 24. エリアチャート（申込の累積推移）
# ===========================================
def chart_application_area():
    """申込の累積推移（エリアチャート、世代別）"""
    weeks = ['9/21', '10/5', '10/19', '11/2', '11/16', '11/30', '12/7', '12/14', '12/21', '12/28']

    # 世代別の累積データ（概算）
    gen1_10 = [8, 16, 25, 38, 52, 80, 115, 170, 205, 209]
    gen11_20 = [5, 12, 20, 30, 45, 75, 105, 155, 183, 187]
    gen21_30 = [3, 5, 8, 10, 14, 22, 35, 70, 100, 104]
    gen31_37 = [1, 2, 2, 2, 3, 6, 9, 20, 30, 33]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(weeks, gen1_10, alpha=0.8, label='1〜10期', color=COLORS['primary'])
    ax.fill_between(weeks, gen11_20, alpha=0.6, label='11〜20期', color=COLORS['secondary'])
    ax.fill_between(weeks, gen21_30, alpha=0.5, label='21〜30期', color='#63B3ED')
    ax.fill_between(weeks, gen31_37, alpha=0.4, label='31〜37期', color='#90CDF4')

    ax.set_xlabel('日付', fontsize=11, fontweight='bold')
    ax.set_ylabel('累積申込者数（名）', fontsize=11, fontweight='bold')
    ax.set_title('世代別 申込推移', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0, 220)
    plt.xticks(rotation=45)

    save_figure(fig, 'application_area.png')

# ===========================================
# 25. ゲージチャート（目標達成率）
# ===========================================
def chart_target_gauge():
    """目標達成率のゲージチャート"""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    targets = [
        ('参加者数', 533, 630, '名'),
        ('収入', 285, 350, '万円'),
        ('料理満足度', 3.34, 4.0, '/5'),
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
    print('Generating charts...')
    setup_style()

    # 基本チャート
    chart_participants_by_class()
    chart_generation_pie()
    chart_satisfaction()
    chart_information_source()
    chart_non_participation_reasons()
    chart_desired_programs()
    chart_fee_tolerance()
    chart_cooperation_willingness()
    chart_involvement_effect()

    # 追加チャート
    chart_application_timeline()
    chart_sponsorship_effect()
    chart_fee_simulation()
    chart_motivation_by_generation()
    chart_motivation_by_department()
    chart_deadline_effect()
    chart_issues_count()

    # 新しい種類のチャート
    chart_generation_radar()
    chart_early_final_scatter()
    chart_revenue_structure()
    chart_satisfaction_heatmap()
    chart_revenue_waterfall()
    chart_opening_conditions()
    chart_bubble_evaluation()
    chart_application_area()
    chart_target_gauge()

    print('All charts generated successfully!')
